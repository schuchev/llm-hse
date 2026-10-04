import os


SEED = 42
DEFAULT_BASE_BATCH_SIZE = 16
LOW_LEARNING_RATE = 1e-4
MID_LEARNING_RATE = 3e-4
HIGH_LEARNING_RATE = 6e-4


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


EXPERIMENT_ORDER = [
    "reference",
    "batch1-lr-low",
    "batch1-lr-high",
    "batch2-lr-low",
    "batch2-lr-high",
    "small-microbatch",
    "scheduler-cosine",
    "no-compile",
    "optimizer-unfused",
    "fp32-tf32",
    "fp32-no-tf32",
]


EXPERIMENT_DESCRIPTIONS = {
    "reference": "Reference: effective batch 2B, mid LR, linear scheduler.",
    "batch1-lr-low": "Effective batch B with low learning rate.",
    "batch1-lr-high": "Effective batch B with high learning rate.",
    "batch2-lr-low": "Effective batch 2B with low learning rate.",
    "batch2-lr-high": "Effective batch 2B with high learning rate.",
    "small-microbatch": "Same effective batch as reference using a smaller microbatch.",
    "scheduler-cosine": "Cosine scheduler ablation against linear reference.",
    "no-compile": "torch_compile=False ablation.",
    "optimizer-unfused": "Unfused AdamW ablation against fused AdamW.",
    "fp32-tf32": "FP32 training with TF32 enabled.",
    "fp32-no-tf32": "FP32 training with TF32 disabled.",
}


def _get_experiment_overrides(base_batch_size):
    if base_batch_size < 4 or base_batch_size % 4 != 0:
        raise ValueError("base_batch_size must be a positive multiple of 4")

    small_microbatch_size = base_batch_size // 4

    return {
        "reference": {},
        "batch1-lr-low": {
            "gradient_accumulation_steps": 1,
            "learning_rate": LOW_LEARNING_RATE,
        },
        "batch1-lr-high": {
            "gradient_accumulation_steps": 1,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "batch2-lr-low": {
            "gradient_accumulation_steps": 2,
            "learning_rate": LOW_LEARNING_RATE,
        },
        "batch2-lr-high": {
            "gradient_accumulation_steps": 2,
            "learning_rate": HIGH_LEARNING_RATE,
        },
        "small-microbatch": {
            "per_device_train_batch_size": small_microbatch_size,
            "gradient_accumulation_steps": 8,
        },
        "scheduler-cosine": {
            "lr_scheduler_type": "cosine",
        },
        "no-compile": {
            "torch_compile": False,
        },
        "optimizer-unfused": {
            "optim": "adamw_torch",
        },
        "fp32-tf32": {
            "per_device_train_batch_size": small_microbatch_size,
            "gradient_accumulation_steps": 8,
            "bf16": False,
            "tf32": True,
        },
        "fp32-no-tf32": {
            "per_device_train_batch_size": small_microbatch_size,
            "gradient_accumulation_steps": 8,
            "bf16": False,
            "tf32": False,
        },
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
        "attention_implementation": (
            "flash_attention_2"
            if training_config["bf16"]
            else "sdpa"
        ),
    }
