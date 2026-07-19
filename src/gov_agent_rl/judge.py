from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUBRIC_VERSION = "gov-expression-v2"
DEFAULT_JUDGE_MODEL = "qwen3.7-max"
DEFAULT_JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

RUBRIC: dict[str, dict[str, Any]] = {
    "clarity": {
        "weight": 0.20,
        "description": "表述清晰、简洁、没有歧义，用户能快速理解当前结论。",
    },
    "reason_completeness": {
        "weight": 0.25,
        "description": "充分说明为什么提交、拒绝或需要补充信息，但不自行判断事实真伪。",
    },
    "actionability": {
        "weight": 0.25,
        "description": "明确说明用户下一步操作、需要补充的内容或后续办理方式。",
    },
    "decision_alignment": {
        "weight": 0.20,
        "description": "回复文字与给定 final_action 在语义上保持一致。",
    },
    "professionalism": {
        "weight": 0.10,
        "description": "语气专业、尊重、克制，不夸大承诺，不使用推诿式表达。",
    },
}


class JudgeCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS judge_cache "
            "(cache_key TEXT PRIMARY KEY, score REAL NOT NULL, payload TEXT NOT NULL)"
        )

    def get(self, key: str) -> tuple[float, dict[str, Any]] | None:
        row = self.connection.execute(
            "SELECT score, payload FROM judge_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return float(row[0]), json.loads(row[1])

    def put(self, key: str, score: float, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO judge_cache(cache_key, score, payload) VALUES (?, ?, ?)",
            (key, score, json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def heuristic_expression_score(message: str) -> float:
    """Deterministic test fallback; never used as a factual verifier."""
    if not message.strip():
        return 0.0
    score = 0.25
    score += 0.25 if len(message) >= 20 else 0.10
    score += 0.25 if any(word in message for word in ("材料", "资格", "风险", "提交")) else 0.0
    score += 0.25 if any(word in message for word in ("请", "建议", "补齐", "转人工")) else 0.0
    return min(1.0, score)


def score_rubric_payload(payload: dict[str, Any]) -> float:
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("judge payload must contain dimensions")
    weighted = 0.0
    for name, spec in RUBRIC.items():
        item = dimensions.get(name)
        raw_score = item.get("score") if isinstance(item, dict) else item
        if raw_score is None or isinstance(raw_score, bool):
            raise ValueError(f"rubric score missing or invalid: {name}={raw_score!r}")
        score = float(raw_score)
        if not 0.0 <= score <= 4.0:
            raise ValueError(f"rubric score out of range: {name}={score}")
        weighted += score * float(spec["weight"])
    return round(weighted / 4.0, 6)


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    return payload


def _rubric_prompt() -> str:
    rubric_lines = []
    for name, spec in RUBRIC.items():
        rubric_lines.append(
            f"- {name}（权重 {spec['weight']:.0%}）：{spec['description']}"
        )
    return (
        "你是政务服务回复质量评审。只评价回复的表达与服务质量，绝不判断政策、资格、材料、"
        "风险或最终动作在事实层面是否正确；这些由确定性 Verifier 负责。\n"
        "按以下维度分别给 0-4 分：0=完全不满足，1=较差，2=基本满足，3=良好，4=优秀。\n"
        + "\n".join(rubric_lines)
        + "\n输出严格 JSON，格式为："
        '{"dimensions":{"clarity":{"score":0,"reason":"..."},'
        '"reason_completeness":{"score":0,"reason":"..."},'
        '"actionability":{"score":0,"reason":"..."},'
        '"decision_alignment":{"score":0,"reason":"..."},'
        '"professionalism":{"score":0,"reason":"..."}},'
        '"summary":"一句话总评"}。每个 score 必须是 0、1、2、3、4 之一，'
        "不得为 null、空字符串或缺失。不要输出总分或 JSON 之外的内容。"
    )


def judge_expression_detailed(
    user_request: str,
    final_action: str,
    message: str,
    cache_path: Path,
) -> tuple[float, dict[str, Any]] | None:
    """Call Qwen through Bailian's OpenAI-compatible API.

    Returns ``None`` when the judge is not configured or unavailable. Set
    ``GOV_JUDGE_REQUIRED=1`` to fail closed instead of silently continuing.
    """
    if not message.strip():
        return 0.0, {
            "dimensions": {
                name: {"score": 0, "reason": "未提供最终回复"} for name in RUBRIC
            },
            "summary": "未提供最终回复",
            "source": "empty-message",
        }

    api_key = os.getenv("GOV_JUDGE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("GOV_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    base_url = os.getenv("GOV_JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL)
    required = os.getenv("GOV_JUDGE_REQUIRED", "0").lower() in {"1", "true", "yes"}
    material = {
        "rubric_version": RUBRIC_VERSION,
        "model": model,
        "base_url": base_url,
        "user_request": user_request,
        "final_action": final_action,
        "message": message,
    }
    key = hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    cache = JudgeCache(cache_path)
    try:
        cached = cache.get(key)
        if cached is not None:
            return cached
        if not api_key:
            if required:
                raise RuntimeError("GOV_JUDGE_REQUIRED=1 but no Bailian API key is configured")
            return None

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("GOV_JUDGE_TIMEOUT", "45")),
            max_retries=int(os.getenv("GOV_JUDGE_MAX_RETRIES", "2")),
        )
        request: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _rubric_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_request": user_request,
                            "final_action": final_action,
                            "assistant_message": message,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "extra_body": {
                "enable_thinking": os.getenv("GOV_JUDGE_ENABLE_THINKING", "0").lower()
                in {"1", "true", "yes"}
            },
        }
        schema_retries = max(0, int(os.getenv("GOV_JUDGE_SCHEMA_RETRIES", "2")))
        payload: dict[str, Any] | None = None
        score: float | None = None
        validation_error: Exception | None = None
        for attempt in range(schema_retries + 1):
            if attempt:
                request["messages"] = [
                    *request["messages"],
                    {
                        "role": "user",
                        "content": (
                            "上一次输出未通过格式校验。请重新输出完整 JSON；五个维度的 "
                            "score 都必须是 0-4 的整数且不能为 null。"
                        ),
                    },
                ]
            try:
                response = client.chat.completions.create(
                    **request,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = client.chat.completions.create(**request)
            try:
                payload = _extract_json(response.choices[0].message.content or "")
                score = score_rubric_payload(payload)
                break
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                validation_error = exc
        if payload is None or score is None:
            raise RuntimeError(
                f"judge returned invalid rubric after {schema_retries + 1} attempts"
            ) from validation_error
        payload["rubric_version"] = RUBRIC_VERSION
        payload["model"] = model
        payload["score"] = score
        cache.put(key, score, payload)
        return score, payload
    except Exception as exc:
        error_log = os.getenv("GOV_JUDGE_ERROR_LOG")
        if error_log:
            path = Path(error_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "model": model,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if required:
            raise
        return None
    finally:
        cache.close()


def judge_expression(
    user_request: str,
    final_action: str,
    message: str,
    cache_path: Path,
) -> float | None:
    result = judge_expression_detailed(
        user_request=user_request,
        final_action=final_action,
        message=message,
        cache_path=cache_path,
    )
    return None if result is None else result[0]
