from __future__ import annotations

import hashlib
import json
from typing import Any


def case_fingerprint(case: Any) -> str:
    if hasattr(case, "model_dump"):
        case = case.model_dump(mode="json")
    canonical = json.dumps(
        case,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
