"""Run reproducible GRPO experiments for ToolACE function calling."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import time
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from rewards import argument_reward, exact_match_reward, format_reward, function_reward


MODEL_NAME = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SEED = 42
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
# This validation subset only selects an RL checkpoint.  The fixed BFCL
# benchmark remains the final quality measurement.
VALIDATION_SAMPLES_FOR_SELECTION = 64


EXPERIMENTS = {
    "smoke_shaped": {
        "reward": "shaped",
        "max_steps": 8,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": False,
        "description": "Eight-step technical GRPO check; not included in the final comparison.",
    },
    "smoke_shaped_vllm": {
        "reward": "shaped",
        "max_steps": 4,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "description": "Four-step vLLM generation speed check; not included in the final comparison.",
    },
    "smoke_shaped_vllm_no_checkpoint": {
        "reward": "shaped",
        "max_steps": 4,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "gradient_checkpointing": False,
        "description": "Four-step vLLM speed check without activation checkpointing.",
    },
    "smoke_shaped_vllm_no_checkpoint_20": {
        "reward": "shaped",
        "max_steps": 20,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 4,
        "gradient_checkpointing": False,
        "description": "Twenty-step end-to-end timing probe for the final G=4 setup.",
    },
    "smoke_shaped_vllm_g8_8": {
        "reward": "shaped",
        "max_steps": 8,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 8,
        "gradient_checkpointing": False,
        "description": "Eight-step timing probe for the final G=8 setup.",
    },
    "smoke_shaped_temp1_20": {
        "reward": "shaped",
        "max_steps": 20,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 4,
        "temperature": 1.0,
        "gradient_checkpointing": False,
        "description": "Twenty-step G=4 probe with more diverse rollout sampling.",
    },
    "rl_exact": {
        "reward": "exact",
        "max_steps": 300,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 4,
        "gradient_checkpointing": False,
        "description": "Sparse binary exact-match reward.",
    },
    "rl_shaped": {
        "reward": "shaped",
        "max_steps": 300,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 4,
        "gradient_checkpointing": False,
        "description": "Partial structural reward; isolates the effect of reward shaping.",
    },
    "rl_shaped_temp1": {
        "reward": "shaped",
        "max_steps": 300,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 4,
        "temperature": 1.0,
        "gradient_checkpointing": False,
        "description": "Shaped reward with a higher rollout temperature for more group diversity.",
    },
    "rl_shaped_temp1_multicall": {
        "reward": "shaped",
        "max_steps": 300,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 4,
        "temperature": 1.0,
        "gradient_checkpointing": False,
        "oversample_multicall": True,
        "description": "Best rollout setup with one extra copy of every multi-call train example.",
    },
    "rl_shaped_g8": {
        "reward": "shaped",
        "max_steps": 300,
        "epochs": 1.0,
        "beta": 0.0,
        "use_vllm": True,
        "num_generations": 8,
        "gradient_checkpointing": False,
        "description": "Shaped reward with eight generations per prompt; isolates group size.",
    },
}


def load_dataset(
    path: Path, tokenizer: AutoTokenizer, *, oversample_multicall: bool = False
) -> Dataset:
    """Render ToolACE messages and schemas into the exact Qwen prompt text."""
    raw_dataset = Dataset.from_json(str(path))

    def render_prompt(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": tokenizer.apply_chat_template(
                row["messages"],
                tools=row["tools"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "target_calls": row["target_calls"],
            "target_type": row["target_type"],
        }

    dataset = raw_dataset.map(
        render_prompt,
        remove_columns=raw_dataset.column_names,
        desc=f"Rendering {path.name}",
    )
    if not oversample_multicall:
        return dataset

    multi_call_dataset = dataset.filter(
        lambda row: len(row["target_calls"]) >= 2,
        desc="Selecting multi-call examples for one extra copy",
    )
    return concatenate_datasets([dataset, multi_call_dataset]).shuffle(seed=SEED)


def reward_functions(reward_name: str) -> list[Any]:
    if reward_name == "exact":
        return [exact_match_reward]
    if reward_name == "shaped":
        # Their sum is 0.10 * format + 0.25 * functions + 0.65 * arguments.
        # Passing them separately makes TRL log every component.
        return [format_reward, function_reward, argument_reward]
    raise ValueError(f"Unknown reward: {reward_name}")


def reward_formula(reward_name: str) -> str:
    if reward_name == "exact":
        return "1 if the complete structured call set matches exactly, else 0"
    return "0.10 * format + 0.25 * function_F1 + 0.65 * argument_score"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    args = parser.parse_args()

    experiment_name = args.experiment
    experiment = EXPERIMENTS[experiment_name]
    output_dir = RESULTS_DIR / experiment_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(
        DATA_DIR / "toolace_train.jsonl",
        tokenizer,
        oversample_multicall=experiment.get("oversample_multicall", False),
    )
    validation_dataset = load_dataset(DATA_DIR / "toolace_validation.jsonl", tokenizer)
    selection_dataset = validation_dataset.shuffle(seed=SEED).select(
        range(min(VALIDATION_SAMPLES_FOR_SELECTION, len(validation_dataset)))
    )

    # Full experiments also use a fixed positive number of optimizer steps.
    # Only explicitly named smoke runs should skip evaluation and checkpoints.
    is_smoke_test = experiment_name.startswith("smoke_")
    config = GRPOConfig(
        output_dir=str(output_dir),
        run_name=experiment_name,
        seed=SEED,
        model_init_kwargs={"dtype": "bfloat16", "revision": MODEL_REVISION},
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=experiment["epochs"],
        max_steps=experiment["max_steps"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_generations=experiment.get("num_generations", 4),
        num_generations_eval=1,
        max_completion_length=512,
        temperature=experiment.get("temperature", 0.7),
        top_p=0.95,
        beta=experiment["beta"],
        use_vllm=experiment["use_vllm"],
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=2560,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=experiment.get("gradient_checkpointing", True),
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no" if is_smoke_test else "steps",
        eval_steps=None if is_smoke_test else 50,
        save_strategy="no" if is_smoke_test else "steps",
        save_steps=500 if is_smoke_test else 50,
        save_total_limit=2,
        load_best_model_at_end=False if is_smoke_test else True,
        metric_for_best_model="reward",
        greater_is_better=True,
    )

    experiment_metadata = {
        "experiment": experiment_name,
        "description": experiment["description"],
        "reward": experiment["reward"],
        "reward_formula": reward_formula(experiment["reward"]),
        "seed": SEED,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset": {
            "train_examples": len(train_dataset),
            "selection_validation_examples": len(selection_dataset),
            "oversample_multicall": experiment.get("oversample_multicall", False),
        },
        "training_args": config.to_dict(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(experiment_metadata, ensure_ascii=False, indent=2, default=str) + "\n"
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_functions(experiment["reward"]),
        args=config,
        train_dataset=train_dataset,
        eval_dataset=None if is_smoke_test else selection_dataset,
        processing_class=tokenizer,
    )

    started_at = time()
    train_result = trainer.train()
    train_seconds = time() - started_at
    eval_metrics = trainer.evaluate() if not is_smoke_test else None

    trainer.save_state()
    trainer.save_model(str(output_dir / "model"))
    (output_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2) + "\n"
    )

    selected_checkpoint = trainer.state.best_model_checkpoint
    selected_step = trainer.state.global_step
    if selected_checkpoint:
        try:
            selected_step = int(Path(selected_checkpoint).name.removeprefix("checkpoint-"))
        except ValueError:
            pass

    summary = {
        "experiment": experiment_name,
        "global_step": trainer.state.global_step,
        "selected_checkpoint": selected_checkpoint,
        "selected_step": selected_step,
        "train_seconds": train_seconds,
        "train_metrics": train_result.metrics,
        "evaluation_metrics": eval_metrics,
        "versions": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "trl": version("trl"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
