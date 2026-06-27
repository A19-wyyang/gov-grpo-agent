# verl GPU Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate verl GRPO jobs that can select physical GPUs such as `4,5,6,7` and automatically configure the matching logical worker count.

**Architecture:** Parse and validate the GPU list at the training-job boundary, then pass normalized IDs into both the generated shell environment and the verl trainer configuration. Keep the existing eight-GPU default when no list is supplied.

**Tech Stack:** Python 3.11, argparse, unittest, Hydra overrides, Bash

---

### Task 1: Parse And Validate GPU IDs

**Files:**
- Modify: `gov_grpo_agent/train_grpo_verl.py`
- Test: `tests/test_train_grpo_verl.py`

- [ ] **Step 1: Write failing parser tests**

```python
from gov_grpo_agent.train_grpo_verl import parse_gpu_ids

def test_parse_gpu_ids_normalizes_valid_list(self):
    self.assertEqual(parse_gpu_ids("4, 5,6,7"), [4, 5, 6, 7])

def test_parse_gpu_ids_rejects_duplicates(self):
    with self.assertRaisesRegex(ValueError, "duplicate"):
        parse_gpu_ids("4,5,4")

def test_parse_gpu_ids_rejects_invalid_values(self):
    for value in ("", "4,,5", "4,x", "-1,4"):
        with self.subTest(value=value), self.assertRaises(ValueError):
            parse_gpu_ids(value)
```

- [ ] **Step 2: Run parser tests and verify they fail**

Run: `python -m unittest tests.test_train_grpo_verl -v`

Expected: FAIL because `parse_gpu_ids` does not exist.

- [ ] **Step 3: Implement the parser**

```python
def parse_gpu_ids(value):
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("GPU list must contain comma-separated non-negative integers")
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("GPU IDs must be integers") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU IDs must be non-negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU list contains duplicate IDs")
    return gpu_ids
```

- [ ] **Step 4: Run parser tests and verify they pass**

Run: `python -m unittest tests.test_train_grpo_verl -v`

Expected: all tests PASS.

### Task 2: Propagate GPU Selection Into Generated Jobs

**Files:**
- Modify: `gov_grpo_agent/train_grpo_verl.py`
- Modify: `gov_grpo_agent/verl_config.py`
- Test: `tests/test_train_grpo_verl.py`
- Test: `tests/test_verl_config.py`

- [ ] **Step 1: Write failing job-generation tests**

Extend the existing job test with `gpu_ids=[4, 5, 6, 7]` and assert:

```python
self.assertEqual(manifest["gpu_ids"], [4, 5, 6, 7])
self.assertEqual(manifest["n_gpus_per_node"], 4)
self.assertIn("trainer.n_gpus_per_node=4", manifest["command"])
script = Path(manifest["run_script"]).read_text(encoding="utf-8")
self.assertIn("export CUDA_VISIBLE_DEVICES=4,5,6,7", script)
```

Add a config test that calls `build_verl_grpo_config(..., n_gpus_per_node=4)` and asserts:

```python
self.assertEqual(config["trainer"]["n_gpus_per_node"], 4)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m unittest tests.test_train_grpo_verl tests.test_verl_config -v`

Expected: FAIL because job preparation does not accept `gpu_ids` and the config builder does not accept `n_gpus_per_node`.

- [ ] **Step 3: Implement configuration and script propagation**

Add `n_gpus_per_node=8` to `build_verl_grpo_config` and use it in the trainer mapping:

```python
"n_gpus_per_node": n_gpus_per_node,
```

Add `gpu_ids=None` to `prepare_verl_training_job`, derive the count, and pass it to the config builder:

```python
n_gpus_per_node = len(gpu_ids) if gpu_ids is not None else 8
```

Pass the IDs to `_write_run_script` and emit the environment only when selected:

```python
if gpu_ids is not None:
    lines.append(f"export CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}")
```

Record both values in the manifest:

```python
"gpu_ids": gpu_ids,
"n_gpus_per_node": n_gpus_per_node,
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `python -m unittest tests.test_train_grpo_verl tests.test_verl_config -v`

Expected: all focused tests PASS.

### Task 3: Add CLI And Documentation

**Files:**
- Modify: `gov_grpo_agent/train_grpo_verl.py`
- Modify: `docs/verl_grpo_training.md`
- Modify: `README.md`
- Test: `tests/test_train_grpo_verl.py`

- [ ] **Step 1: Add a failing CLI argument test**

Patch `prepare_verl_training_job`, call `main` with `--gpus 4,5,6,7`, and assert the mock receives:

```python
gpu_ids=[4, 5, 6, 7]
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run: `python -m unittest tests.test_train_grpo_verl -v`

Expected: FAIL because argparse does not define `--gpus`.

- [ ] **Step 3: Add CLI wiring**

```python
parser.add_argument("--gpus", help="Comma-separated physical GPU IDs, for example 4,5,6,7")
gpu_ids = parse_gpu_ids(args.gpus) if args.gpus is not None else None
```

Pass `gpu_ids=gpu_ids` into `prepare_verl_training_job`.

- [ ] **Step 4: Update server commands**

Document the mixed-memory server command:

```bash
python -m gov_grpo_agent.train_grpo_verl \
  --input-jsonl artifacts/grpo_train/qwen3_grpo_train_sampled.jsonl \
  --work-dir artifacts/verl_grpo_qwen3_8b \
  --model-path Qwen/Qwen3-8B \
  --n-rollout 4 \
  --total-epochs 1 \
  --gpus 4,5,6,7
```

- [ ] **Step 5: Run full verification**

Run: `python -m unittest discover -s tests -v`

Expected: 0 failures and 0 errors.

Run: `git diff --check`

Expected: no whitespace errors.
