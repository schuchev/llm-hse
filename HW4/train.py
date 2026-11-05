import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from experiment_configs import MAX_LENGTH, MODEL_NAME, RESULTS_DIR, SEED, get_experiment


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def load_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    if config["method"] == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["r"],
            lora_alpha=config["alpha"],
            lora_dropout=config["dropout"],
            target_modules=config["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    return model


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    config = get_experiment(args.experiment)
    output_dir = RESULTS_DIR / args.experiment
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
    validation_dataset = load_dataset("json", data_files="data/validation.jsonl", split="train")
    model = load_model(config)
    total_params, trainable_params = count_parameters(model)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=config["logging_steps"],
        report_to="none",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )

    save_json(output_dir / "config.json", config)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started_at
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "model"))

    metrics = {
        "experiment": args.experiment,
        "method": config["method"],
        "train_seconds": train_seconds,
        "global_step": trainer.state.global_step,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "log_history.json", trainer.state.log_history)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
