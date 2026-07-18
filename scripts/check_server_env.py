from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path


MODULES = [
    "torch",
    "verl",
    "vllm",
    "transformers",
    "peft",
    "ray",
    "datasets",
    "bitsandbytes",
    "pydantic",
]


def main() -> None:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "modules": {},
    }
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = getattr(module, "__version__", "installed")  # type: ignore[index]
        except Exception as exc:
            result["modules"][name] = f"ERROR: {exc}"  # type: ignore[index]

    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 2**30,
                        2,
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception:
        pass

    result["project"] = {
        "cwd": str(Path.cwd()),
        "tool_config": Path("configs/tools/government_service.yaml").exists(),
        "train_parquet": Path("data/processed/train.parquet").exists(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
