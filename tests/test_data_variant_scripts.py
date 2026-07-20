from pathlib import Path


def test_diverse_v2_training_is_isolated_from_legacy_paths():
    root = Path(__file__).resolve().parents[1]
    sft = (root / "scripts/train_sft_diverse_v2.sh").read_text(encoding="utf-8")
    grpo = (root / "scripts/train_grpo_diverse_v2.sh").read_text(encoding="utf-8")
    assert "data/processed_v2/train.sft.jsonl" in sft
    assert "sft-qwen3-8b-diverse-v2" in sft
    assert "data/processed_v2/train.parquet" in grpo
    assert "data/processed_v2/validation.parquet" in grpo
    assert "government_service.yaml" in grpo
    assert "government_service_h10.yaml" not in grpo
    assert "reward_v2_env.sh" in grpo


def test_evaluation_accepts_an_explicit_dataset_version():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/evaluate_grpo.sh").read_text(encoding="utf-8")
    assert 'EVAL_DATA_DIR="${EVAL_DATA_DIR:-' in script
    assert 'EVAL_CASES_FILE="${EVAL_CASES_FILE:-' in script


def test_diverse_comparison_uses_one_fixed_v2_test_for_both_models():
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts/compare_diverse_v2_on_fixed_test.sh"
    ).read_text(encoding="utf-8")
    assert script.count('EVAL_DATA_DIR="${PROJECT_DIR}/data/processed_v2"') == 2
    assert script.count("EVAL_ROLLOUT_N=4") == 2
    assert "data/processed_v2/test.cases.jsonl" in script
    assert "--allow-missing-case-fingerprint" not in script
    assert "decide_grpo_promotion.py" in script


def test_curriculum_training_uses_validation_only_and_keeps_ablation_fixed():
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts/train_grpo_curriculum_v3.sh"
    ).read_text(encoding="utf-8")
    assert "--metrics-split validation" in script
    assert "rescore_rollouts.py" in script
    assert "validation.cases.jsonl" in script
    assert "baseline_validation_common_reward.jsonl" in script
    assert "processed_v2/train.jsonl" in script
    assert "processed_v2/validation.parquet" in script
    assert "test.jsonl" not in script
    assert "TRAIN_BATCH_SIZE:-16" in script
    assert "ROLLOUT_N:-4" in script
    assert "TOTAL_TRAINING_STEPS:-25" in script
    assert "government_service_h10.yaml" not in script


def test_curriculum_comparison_selects_on_validation_then_uses_fixed_v2_test():
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts/compare_curriculum_v3_on_fixed_test.sh"
    ).read_text(encoding="utf-8")
    assert script.count("select_best_grpo_checkpoint.py") == 1
    assert "baseline_validation_selection" in script
    assert "candidate_validation_selection" in script
    assert script.count('EVAL_ROLLOUT_N=4') == 1
    assert "data/processed_v2/test.cases.jsonl" in script
    assert "--allow-missing-case-fingerprint" not in script
    assert "decide_grpo_promotion.py" in script


def test_grpo_seed_controls_all_relevant_verl_rngs_and_manifest():
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts/train_grpo.sh").read_text(encoding="utf-8")
    manifest = (
        root / "scripts/write_run_manifest.py"
    ).read_text(encoding="utf-8")
    for setting in (
        'data.seed="${TRAIN_SEED}"',
        'actor_rollout_ref.actor.data_loader_seed="${TRAIN_SEED}"',
        'actor_rollout_ref.actor.fsdp_config.seed="${TRAIN_SEED}"',
        'actor_rollout_ref.ref.fsdp_config.seed="${TRAIN_SEED}"',
        'actor_rollout_ref.rollout.seed="${ROLLOUT_SEED}"',
    ):
        assert setting in train
    assert '--seed "${TRAIN_SEED}"' in train
    assert '"seed": args.seed' in manifest
    assert 'actor_rollout_ref.rollout.val_kwargs.n="${EVAL_ROLLOUT_N}"' in train
    assert (
        'actor_rollout_ref.rollout.val_kwargs.temperature="${EVAL_TEMPERATURE}"'
        in train
    )
    assert (
        'actor_rollout_ref.rollout.val_kwargs.do_sample="${EVAL_DO_SAMPLE}"'
        in train
    )


def test_fixed_test_uses_reproducible_stochastic_rollouts():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/evaluate_grpo.sh").read_text(encoding="utf-8")
    assert 'EVAL_SEED="${EVAL_SEED:-20260719}"' in script
    assert 'ROLLOUT_SEED="${ROLLOUT_SEED:-${EVAL_SEED}}"' in script
    assert 'EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"' in script
    assert 'EVAL_TOP_P="${EVAL_TOP_P:-0.95}"' in script
    assert 'EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-True}"' in script


def test_multiseed_runner_requires_three_seeds_and_pairs_each_seed():
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts/run_curriculum_multiseed.sh"
    ).read_text(encoding="utf-8")
    assert 'SEEDS:-42 43 44' in script
    assert "at least three seeds" in script
    assert "train_grpo_diverse_v2.sh" in script
    assert "train_grpo_curriculum_v3.sh" in script
    assert "compare_curriculum_v3_on_fixed_test.sh" in script
    assert "aggregate_seed_comparisons.py" in script


def test_turn_balanced_sft_is_an_isolated_ablation():
    root = Path(__file__).resolve().parents[1]
    sft = (
        root / "scripts/train_sft_turn_balanced.sh"
    ).read_text(encoding="utf-8")
    grpo = (
        root / "scripts/train_grpo_turn_balanced_sft.sh"
    ).read_text(encoding="utf-8")
    comparison = (
        root / "scripts/compare_turn_balanced_sft_on_fixed_test.sh"
    ).read_text(encoding="utf-8")
    assert "--turn-balanced-loss" in sft
    assert "processed_v2/train.sft.jsonl" in sft
    assert "diverse-v2-turn-balanced/final_adapter" in grpo
    assert "processed_v2/train.parquet" in grpo
    assert "TRAIN_BATCH_SIZE:-16" in grpo
    assert "ROLLOUT_N:-4" in grpo
    assert "TOTAL_TRAINING_STEPS:-25" in grpo
    assert "government_service_h10.yaml" not in grpo
    assert "BASELINE_SFT_ADAPTER" in comparison
    assert "CANDIDATE_SFT_ADAPTER" in comparison


def test_formal_sft_restores_best_validation_checkpoint_and_exports_plots():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/train_sft.py").read_text(encoding="utf-8")
    assert "load_best_model_at_end=not args.smoke" in script
    assert 'metric_for_best_model="eval_loss"' in script
    assert "greater_is_better=False" in script
    assert "best_model_checkpoint is None" in script
    assert "sft_training_metrics.png" in script
    assert "sft_scenario_metrics.png" in script


def test_sequence_balanced_grpo_changes_only_loss_aggregation():
    root = Path(__file__).resolve().parents[1]
    train = (
        root / "scripts/train_grpo_sequence_balanced.sh"
    ).read_text(encoding="utf-8")
    comparison = (
        root / "scripts/compare_sequence_balanced_on_fixed_test.sh"
    ).read_text(encoding="utf-8")
    assert 'LOSS_AGG_MODE="seq-mean-token-mean"' in train
    assert "processed_v2/train.parquet" in train
    assert "TRAIN_BATCH_SIZE:-16" in train
    assert "ROLLOUT_N:-4" in train
    assert "TOTAL_TRAINING_STEPS:-25" in train
    assert "government_service_h10.yaml" not in train
    assert "compare_curriculum_v3_on_fixed_test.sh" in comparison


def test_grpo_loss_aggregation_is_explicit_and_manifested():
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts/train_grpo.sh").read_text(encoding="utf-8")
    manifest = (
        root / "scripts/write_run_manifest.py"
    ).read_text(encoding="utf-8")
    assert 'LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"' in train
    assert (
        'actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"'
        in train
    )
    assert '--loss-agg-mode "${LOSS_AGG_MODE}"' in train
    assert '"loss_agg_mode": args.loss_agg_mode' in manifest


def test_grpo_group_filtering_is_explicit_and_manifested():
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts/train_grpo.sh").read_text(encoding="utf-8")
    manifest = (
        root / "scripts/write_run_manifest.py"
    ).read_text(encoding="utf-8")
    for setting in (
        'GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"',
        'FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"',
        'algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}"',
        'algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}"',
        'algorithm.filter_groups.max_num_gen_batches="${FILTER_MAX_GEN_BATCHES}"',
        'data.gen_batch_size="${GEN_BATCH_SIZE}"',
    ):
        assert setting in train
    for setting in (
        '--gen-batch-size "${GEN_BATCH_SIZE}"',
        '--filter-groups-enable "${FILTER_GROUPS_ENABLE}"',
        '--filter-groups-metric "${FILTER_GROUPS_METRIC}"',
        '--filter-max-gen-batches "${FILTER_MAX_GEN_BATCHES}"',
    ):
        assert setting in train
    for field in (
        '"gen_batch_size": args.gen_batch_size',
        '"filter_groups_enable": (',
        "args.filter_groups_enable.lower()",
        '"filter_groups_metric": args.filter_groups_metric',
        '"filter_max_gen_batches": args.filter_max_gen_batches',
    ):
        assert field in manifest


def test_informative_group_ablation_changes_only_dynamic_sampling():
    root = Path(__file__).resolve().parents[1]
    train = (
        root / "scripts/train_grpo_informative_groups.sh"
    ).read_text(encoding="utf-8")
    comparison = (
        root / "scripts/compare_informative_groups_on_fixed_test.sh"
    ).read_text(encoding="utf-8")
    for setting in (
        "data/processed_v2/train.parquet",
        "data/processed_v2/validation.parquet",
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"',
        'GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-24}"',
        'ROLLOUT_N="${ROLLOUT_N:-4}"',
        "FILTER_GROUPS_ENABLE=True",
        "FILTER_GROUPS_METRIC=score",
        'FILTER_MAX_GEN_BATCHES="${FILTER_MAX_GEN_BATCHES:-3}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"',
        "government_service.yaml",
        "LOSS_AGG_MODE=token-mean",
    ):
        assert setting in train
    assert "government_service_h10.yaml" not in train
    assert "compare_curriculum_v3_on_fixed_test.sh" in comparison


def test_grpo_asymmetric_clipping_is_explicit_and_manifested():
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts/train_grpo.sh").read_text(encoding="utf-8")
    manifest = (
        root / "scripts/write_run_manifest.py"
    ).read_text(encoding="utf-8")
    for setting in (
        'CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-${CLIP_RATIO}}"',
        'CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-${CLIP_RATIO}}"',
        'actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}"',
        'actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"',
        '--clip-ratio-low "${CLIP_RATIO_LOW}"',
        '--clip-ratio-high "${CLIP_RATIO_HIGH}"',
    ):
        assert setting in train
    assert '"clip_ratio_low": args.clip_ratio_low' in manifest
    assert '"clip_ratio_high": args.clip_ratio_high' in manifest


def test_clip_higher_ablation_changes_only_the_positive_clip_bound():
    root = Path(__file__).resolve().parents[1]
    train = (
        root / "scripts/train_grpo_clip_higher.sh"
    ).read_text(encoding="utf-8")
    comparison = (
        root / "scripts/compare_clip_higher_on_fixed_test.sh"
    ).read_text(encoding="utf-8")
    for setting in (
        "data/processed_v2/train.parquet",
        "data/processed_v2/validation.parquet",
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"',
        'GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"',
        'ROLLOUT_N="${ROLLOUT_N:-4}"',
        "FILTER_GROUPS_ENABLE=False",
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"',
        "government_service.yaml",
        "LOSS_AGG_MODE=token-mean",
        "CLIP_RATIO=0.2",
        "CLIP_RATIO_LOW=0.2",
        "CLIP_RATIO_HIGH=0.28",
    ):
        assert setting in train
    assert "government_service_h10.yaml" not in train
    assert "compare_curriculum_v3_on_fixed_test.sh" in comparison


def test_generic_multiseed_runner_is_screened_and_seed_paired():
    root = Path(__file__).resolve().parents[1]
    runner = (
        root / "scripts/run_ablation_multiseed.sh"
    ).read_text(encoding="utf-8")
    clip_wrapper = (
        root / "scripts/run_clip_higher_multiseed.sh"
    ).read_text(encoding="utf-8")
    group_wrapper = (
        root / "scripts/run_informative_groups_multiseed.sh"
    ).read_text(encoding="utf-8")
    assert 'SEEDS:-42 43 44' in runner
    assert "at least three seeds" in runner
    assert "SCREENING_DECISION_FILE" in runner
    assert '"reject" || "${screening_decision}" == "invalid"' in runner
    assert 'TRAIN_SEED="${seed}"' in runner
    assert 'ROLLOUT_SEED="${seed}"' in runner
    assert "aggregate_seed_comparisons.py" in runner
    assert "--title" in runner
    assert "train_grpo_clip_higher.sh" in clip_wrapper
    assert 'CANDIDATE_SLUG="cliphigher"' in clip_wrapper
    assert "train_grpo_informative_groups.sh" in group_wrapper
    assert 'CANDIDATE_SLUG="informative_groups"' in group_wrapper
