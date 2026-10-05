import os


SEED = 42
DEFAULT_BASE_BATCH_SIZE = 16
MID_LEARNING_RATE = 3e-4
HIGH_LEARNING_RATE = 6e-4
GRID_LOW_LEARNING_RATE = 1.5e-4
GRID_EXTENDED_LEARNING_RATE = 1.2e-3


COMMON_TRAINING_CONFIG = {
    "optim": "adamw_torch_fused",
    "num_train_epochs": 1,
    "per_device_train_batch_size": DEFAULT_BASE_BATCH_SIZE,
    "save_steps": 100,
    "save_total_limit": 2,
    "learning_rate": MID_LEARNING_RATE,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "logging_steps": 1,
    "logging_first_step": True,
    "eval_steps": 100,
    "eval_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 2,
    "dataloader_num_workers": 4,
    "torch_compile": True,
    "skip_memory_metrics": False,
    "seed": SEED,
    "data_seed": SEED,
}


STAGE_A_TRAINING_CONFIG = {
    # Shared protocol for the completed Stage A-F experiments. The planned
    # horizon exceeds 15 minutes; TimeoutCallback stops the actual training.
    "num_train_epochs": 0.032,
    "save_strategy": "no",
    # TimeoutCallback requests one final checkpoint. Optimizer and scheduler
    # states are not needed for these non-resumable runs.
    "save_only_model": True,
    "eval_strategy": "no",
    "load_best_model_at_end": False,
    "warmup_steps": 0,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
}


SDPA_EXPERIMENTS = {
    "precision_bf16_sdpa",
    "precision_fp32_tf32",
    "precision_fp32_no_tf32",
}


def get_attention_implementation(experiment_name, training_config):
    if experiment_name in SDPA_EXPERIMENTS:
        return "sdpa"
    return "flash_attention_2" if training_config["bf16"] else "sdpa"


EXPERIMENT_ORDER = [
    "reference",
    "grid_b16_lr1p5e4",
    "grid_b16_lr3e4",
    "grid_b16_lr6e4",
    "grid_b32_lr1p5e4",
    "grid_b32_lr3e4",
    "grid_b32_lr6e4",
    "grid_b64_lr1p5e4",
    "grid_b64_lr3e4",
    "grid_b64_lr6e4",
    "grid_b16_lr1p2e3",
    "grid_b32_lr1p2e3",
    "grid_b64_lr1p2e3",
    "ga_b8x2",
    "ga_b4x4",
    "scheduler_cosine",
    "compile_off",
    "optimizer_unfused",
    "precision_bf16_sdpa",
    "precision_fp32_tf32",
    "precision_fp32_no_tf32",
]


EXPERIMENT_DESCRIPTIONS = {
    "grid_b16_lr1p5e4": "Stage A grid: batch 16, learning rate 1.5e-4.",
    "grid_b16_lr3e4": "Stage A grid: batch 16, learning rate 3e-4.",
    "grid_b16_lr6e4": "Stage A grid: batch 16, learning rate 6e-4.",
    "grid_b32_lr1p5e4": "Stage A grid: batch 32, learning rate 1.5e-4.",
    "grid_b32_lr3e4": "Stage A grid: batch 32, learning rate 3e-4.",
    "grid_b32_lr6e4": "Stage A grid: batch 32, learning rate 6e-4.",
    "grid_b64_lr1p5e4": "Stage A grid: batch 64, learning rate 1.5e-4.",
    "grid_b64_lr3e4": "Stage A grid: batch 64, learning rate 3e-4.",
    "grid_b64_lr6e4": "Stage A grid: batch 64, learning rate 6e-4.",
    "grid_b16_lr1p2e3": "Stage A extension: batch 16, learning rate 1.2e-3.",
    "grid_b32_lr1p2e3": "Stage A extension: batch 32, learning rate 1.2e-3.",
    "grid_b64_lr1p2e3": "Stage A extension: batch 64, learning rate 1.2e-3.",
    "ga_b8x2": "Stage B: microbatch 8, accumulation 2, effective batch 16.",
    "ga_b4x4": "Stage B: microbatch 4, accumulation 4, effective batch 16.",
    "scheduler_cosine": "Stage C: cosine scheduler with batch 16, accumulation 1, learning rate 6e-4.",
    "compile_off": "Stage D: disable torch_compile with batch 16, accumulation 1, learning rate 6e-4.",
    "optimizer_unfused": "Stage E: unfused AdamW with batch 16, accumulation 1, learning rate 6e-4.",
    "precision_bf16_sdpa": "Stage F: BF16 with SDPA, batch 16, accumulation 1, learning rate 6e-4.",
    "precision_fp32_tf32": "Stage F: FP32 with SDPA and TF32 enabled, batch 16, accumulation 1, learning rate 6e-4.",
    "precision_fp32_no_tf32": "Stage F: FP32 with SDPA and TF32 disabled, batch 16, accumulation 1, learning rate 6e-4.",
    "reference": "Pilot run: microbatch 16, accumulation 2, learning rate 3e-4.",
}


def _get_experiment_overrides(base_batch_size):
    if base_batch_size < 4 or base_batch_size % 4 != 0:
        raise ValueError("base_batch_size must be a positive multiple of 4")

    return {
        "grid_b16_lr1p5e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": GRID_LOW_LEARNING_RATE,
        },
        "grid_b16_lr3e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": MID_LEARNING_RATE,
        },
        "grid_b16_lr6e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "grid_b32_lr1p5e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": GRID_LOW_LEARNING_RATE,
        },
        "grid_b32_lr3e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": MID_LEARNING_RATE,
        },
        "grid_b32_lr6e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "grid_b64_lr1p5e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "learning_rate": GRID_LOW_LEARNING_RATE,
        },
        "grid_b64_lr3e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "learning_rate": MID_LEARNING_RATE,
        },
        "grid_b64_lr6e4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "grid_b16_lr1p2e3": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": GRID_EXTENDED_LEARNING_RATE,
        },
        "grid_b32_lr1p2e3": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": GRID_EXTENDED_LEARNING_RATE,
        },
        "grid_b64_lr1p2e3": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "learning_rate": GRID_EXTENDED_LEARNING_RATE,
        },
        "ga_b8x2": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "ga_b4x4": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "scheduler_cosine": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
            "lr_scheduler_type": "cosine",
        },
        "compile_off": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
            "torch_compile": False,
        },
        "optimizer_unfused": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
            "optim": "adamw_torch",
        },
        "precision_bf16_sdpa": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "precision_fp32_tf32": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
            "bf16": False,
            "tf32": True,
        },
        "precision_fp32_no_tf32": {
            **STAGE_A_TRAINING_CONFIG,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
            "bf16": False,
            "tf32": False,
        },
        "reference": {},
    }


def build_training_config(
    experiment_name,
    base_batch_size=DEFAULT_BASE_BATCH_SIZE,
    results_dir="./results",
    tracker="tensorboard",
):
    overrides = _get_experiment_overrides(base_batch_size)

    if experiment_name not in overrides:
        available = ", ".join(EXPERIMENT_ORDER)
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. Available: {available}"
        )

    config = COMMON_TRAINING_CONFIG.copy()
    config["per_device_train_batch_size"] = base_batch_size
    config.update(overrides[experiment_name])
    experiment_dir = os.path.join(results_dir, experiment_name)
    config["output_dir"] = experiment_dir
    config["logging_dir"] = os.path.join(experiment_dir, "tensorboard")
    config["run_name"] = experiment_name
    config["report_to"] = tracker
    return config


def get_experiment_metadata(experiment_name, training_config):
    microbatch_size = training_config["per_device_train_batch_size"]
    accumulation_steps = training_config["gradient_accumulation_steps"]

    return {
        "experiment_name": experiment_name,
        "description": EXPERIMENT_DESCRIPTIONS[experiment_name],
        "seed": training_config["seed"],
        "per_device_train_batch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": microbatch_size * accumulation_steps,
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "optim": training_config["optim"],
        "torch_compile": training_config["torch_compile"],
        "bf16": training_config["bf16"],
        "tf32": training_config["tf32"],
        "attention_implementation": get_attention_implementation(
            experiment_name,
            training_config,
        ),
    }
