from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


RUBRIC_VERSION = "gov-expression-v1"


class JudgeCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS judge_cache "
            "(cache_key TEXT PRIMARY KEY, score REAL NOT NULL, payload TEXT NOT NULL)"
        )

    def get(self, key: str) -> float | None:
        row = self.connection.execute(
            "SELECT score FROM judge_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return None if row is None else float(row[0])

    def put(self, key: str, score: float, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO judge_cache(cache_key, score, payload) VALUES (?, ?, ?)",
            (key, score, json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.commit()


def heuristic_expression_score(message: str) -> float:
    """Deterministic fallback for tests; never used as a factual verifier."""
    if not message.strip():
        return 0.0
    score = 0.25
    score += 0.25 if len(message) >= 20 else 0.10
    score += 0.25 if any(word in message for word in ("材料", "资格", "风险", "提交")) else 0.0
    score += 0.25 if any(word in message for word in ("请", "建议", "补齐", "转人工")) else 0.0
    return min(1.0, score)


def judge_expression(
    user_request: str,
    final_action: str,
    message: str,
    cache_path: Path,
) -> float | None:
    """Call an OpenAI-compatible judge. Returns None on unavailable API."""
    material = {
        "rubric_version": RUBRIC_VERSION,
        "model": os.getenv("GOV_JUDGE_MODEL", ""),
        "user_request": user_request,
        "final_action": final_action,
        "message": message,
    }
    key = hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    cache = JudgeCache(cache_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not os.getenv("GOV_JUDGE_API_KEY") or not os.getenv("GOV_JUDGE_MODEL"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["GOV_JUDGE_API_KEY"],
            base_url=os.getenv("GOV_JUDGE_BASE_URL") or None,
            timeout=20,
            max_retries=2,
        )
        response = client.chat.completions.create(
            model=os.environ["GOV_JUDGE_MODEL"],
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "只评价政务回复的清晰度、理由完整性和可执行性，不评价事实是否正确。"
                        "输出 JSON：{\"score\": 0到1之间的数字, \"reason\": \"简短理由\"}。"
                    ),
                },
                {"role": "user", "content": json.dumps(material, ensure_ascii=False)},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        score = max(0.0, min(1.0, float(payload["score"])))
    except Exception:
        return None
    cache.put(key, score, payload)
    return score
