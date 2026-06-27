# verl GPU Selection Design

## Goal

Allow a generated verl GRPO job to target an explicit subset of physical GPUs.
For the current mixed-memory server, the intended selection is GPUs 4, 5, 6,
and 7, which each provide 48 GB of memory.

## Interface

`python -m gov_grpo_agent.train_grpo_verl` accepts an optional `--gpus`
argument containing a comma-separated list such as `4,5,6,7`.

When provided:

- the generated shell script exports `CUDA_VISIBLE_DEVICES=4,5,6,7`;
- `trainer.n_gpus_per_node` is set to the number of selected devices;
- the manifest records the physical GPU IDs and logical GPU count.

When omitted, existing behavior remains unchanged and the job uses the
configured default of eight GPUs.

## Validation

GPU IDs must be non-negative integers without duplicates. Empty entries,
invalid integers, and duplicate IDs fail during job preparation with a clear
error before any parquet or training process is launched.

## Generated Job

verl sees the selected devices as logical CUDA devices starting at zero. For
example, physical GPUs `4,5,6,7` become logical devices `0,1,2,3`, while
`trainer.n_gpus_per_node` is set to `4`.

## Tests

Tests cover parsing and validation of GPU lists, propagation into the generated
Hydra overrides, the `CUDA_VISIBLE_DEVICES` export in the shell script, manifest
metadata, and backward-compatible behavior when `--gpus` is omitted.
