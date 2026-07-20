import json

import pytest

from scripts.write_run_manifest import build_manifest, write_or_verify_manifest


def _fixture(tmp_path):
    data = tmp_path / "data"
    adapter = tmp_path / "adapter"
    data.mkdir()
    adapter.mkdir()
    train = data / "train.parquet"
    validation = data / "validation.parquet"
    tool = tmp_path / "tool.yaml"
    train.write_bytes(b"train-v1")
    validation.write_bytes(b"validation-v1")
    tool.write_text("max_steps: 8\n", encoding="utf-8")
    (adapter / "adapter.safetensors").write_bytes(b"adapter-v1")
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_variant": "legacy_v1",
                "case_fingerprint_sha256": "dataset-fingerprint",
            }
        ),
        encoding="utf-8",
    )
    return train, validation, tool, adapter


def _manifest(tmp_path):
    train, validation, tool, adapter = _fixture(tmp_path)
    return build_manifest(
        project_dir=tmp_path,
        experiment="experiment",
        train_file=train,
        val_file=validation,
        model="Qwen/Qwen3-8B",
        sft_adapter=adapter,
        tool_config=tool,
        training={"rollout_n": 4},
    )


def test_run_manifest_is_idempotent(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "run_manifest.json"
    assert write_or_verify_manifest(output, manifest) == "created"
    assert write_or_verify_manifest(output, manifest) == "verified"


def test_run_manifest_rejects_configuration_drift(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "run_manifest.json"
    write_or_verify_manifest(output, manifest)
    changed = json.loads(json.dumps(manifest))
    changed["training"]["rollout_n"] = 8
    with pytest.raises(ValueError, match="run manifest mismatch"):
        write_or_verify_manifest(output, changed)
    candidate = tmp_path / "run_manifest.candidate.json"
    assert candidate.is_file()
    assert json.loads(candidate.read_text(encoding="utf-8"))["training"][
        "rollout_n"
    ] == 8
