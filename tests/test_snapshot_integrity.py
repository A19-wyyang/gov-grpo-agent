import json

from scripts.check_grpo_snapshot import check_snapshot


def _write_checkpoint(root, world_size=2):
    actor = root / "actor"
    (actor / "huggingface").mkdir(parents=True)
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"x")
    for path in (
        root / "data.pt",
        actor / "fsdp_config.json",
        actor / "lora_train_meta.json",
        actor / "huggingface" / "config.json",
        actor / "huggingface" / "tokenizer.json",
    ):
        path.write_text("{}", encoding="utf-8")


def test_snapshot_requires_complete_validation_and_checkpoint(tmp_path):
    validation = tmp_path / "25.jsonl"
    checkpoint = tmp_path / "global_step_25"
    _write_checkpoint(checkpoint)
    validation.write_text(
        "".join(
            json.dumps({"step": 25, "case_id": f"case-{index}"}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    complete = check_snapshot(validation, checkpoint, 25, 3, 2)
    assert complete["ok"]
    assert complete["validation_records"] == 3
    assert complete["unique_cases"] == 3

    validation.write_text(
        json.dumps({"step": 25, "case_id": "case-0"}) + "\n",
        encoding="utf-8",
    )
    partial = check_snapshot(validation, checkpoint, 25, 3, 2)
    assert not partial["ok"]
    assert any("record count" in error for error in partial["errors"])


def test_snapshot_rejects_missing_model_shard(tmp_path):
    validation = tmp_path / "25.jsonl"
    checkpoint = tmp_path / "global_step_25"
    _write_checkpoint(checkpoint)
    validation.write_text(
        json.dumps({"step": 25, "case_id": "case-0"}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "actor" / "model_world_size_2_rank_1.pt").unlink()
    result = check_snapshot(validation, checkpoint, 25, 1, 2)
    assert not result["ok"]
    assert result["model_shards"] == 1


def test_snapshot_rejects_invalid_step_and_files_still_being_written(tmp_path):
    validation = tmp_path / "25.jsonl"
    checkpoint = tmp_path / "global_step_25"
    _write_checkpoint(checkpoint)
    validation.write_text(
        json.dumps({"step": "not-a-step", "case_id": "case-0"}) + "\n",
        encoding="utf-8",
    )
    result = check_snapshot(
        validation,
        checkpoint,
        step=25,
        expected_cases=1,
        world_size=2,
        min_age_seconds=30,
    )
    assert not result["ok"]
    assert any("invalid step" in error for error in result["errors"])
    assert any("too fresh" in error for error in result["errors"])
