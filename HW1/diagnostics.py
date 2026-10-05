import argparse
import math
import os
import statistics
import time

import torch
from transformers import (
    set_seed,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from experiment_configs import SEED
from your_solution import (
    MAX_LENGTH,
    VALIDATION_SIZE,
    create_model,
    generate_samples,
    load_tokenized_dataset,
    prepare_tokenizer,
    save_json,
)


class StepTimingCallback(TrainerCallback):
    """Measure completed optimizer-step times during a batch-size probe."""
    def __init__(self):
        self.step_end_times = []

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_end_times.append(time.perf_counter())
        return control

    def median_step_seconds(self):
        if len(self.step_end_times) < 2:
            return None

        step_durations = [
            current - previous
            for previous, current in zip(
                self.step_end_times,
                self.step_end_times[1:],
            )
        ]
        return statistics.median(step_durations)


def add_perplexity(metrics):
    eval_loss = metrics.get("eval_loss")
    if eval_loss is None:
        return

    try:
        metrics["perplexity"] = math.exp(eval_loss)
    except OverflowError:
        metrics["perplexity"] = None


def is_cuda_out_of_memory(error):
    """Detect a CUDA OOM, including errors wrapped by torch.compile."""
    current_error = error
    visited_errors = set()

    while current_error is not None and id(current_error) not in visited_errors:
        visited_errors.add(id(current_error))
        if isinstance(current_error, torch.cuda.OutOfMemoryError):
            return True
        if "CUDA out of memory" in str(current_error):
            return True
        current_error = current_error.__cause__ or current_error.__context__

    return False


def evaluate_untrained(results_dir="./results"):
    """Evaluate and generate from a freshly initialized model."""
    output_dir = os.path.join(results_dir, "untrained")
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            f"Result directory is not empty: {output_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)
    set_seed(SEED)
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    eval_dataset = dataset.select(range(VALIDATION_SIZE))
    model = create_model(tokenizer)

    config = {
        "model_state": "untrained",
        "seed": SEED,
        "validation_size": VALIDATION_SIZE,
        "per_device_eval_batch_size": 8,
        "bf16": True,
        "tf32": True,
        "torch_compile": False,
        "attention_implementation": "flash_attention_2",
    }
    save_json(config, os.path.join(output_dir, "config.json"))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        report_to="none",
        skip_memory_metrics=False,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Evaluating the untrained model...")
    eval_results = trainer.evaluate()
    add_perplexity(eval_results)
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    generations = generate_samples(trainer.model, tokenizer)
    save_json(
        generations,
        os.path.join(output_dir, "generations.json"),
    )
    save_json(
        {
            "model_state": "untrained",
            "seed": SEED,
            "eval_loss": eval_results.get("eval_loss"),
            "perplexity": eval_results.get("perplexity"),
            "eval_metrics": eval_results,
        },
        os.path.join(output_dir, "summary.json"),
    )


def probe_batch_size(batch_size, steps=5, results_dir="./results"):
    """Run a few optimizer steps to test one microbatch size."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    set_seed(SEED)
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    first_train_index = VALIDATION_SIZE
    sample_count = batch_size * steps
    probe_dataset = dataset.select(
        range(first_train_index, first_train_index + sample_count)
    )
    model = create_model(tokenizer)
    timing_callback = StepTimingCallback()

    training_args = TrainingArguments(
        output_dir=f"/tmp/hw1-batch-probe-{batch_size}",
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        torch_compile=True,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        logging_first_step=True,
        dataloader_num_workers=4,
        report_to="none",
        skip_memory_metrics=True,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=probe_dataset,
        processing_class=tokenizer,
        callbacks=[timing_callback],
    )

    result_dir = os.path.join(results_dir, "batch-probes")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"batch-{batch_size}.json")
    torch.cuda.reset_peak_memory_stats()

    try:
        trainer.train()
    except Exception as error:
        if not is_cuda_out_of_memory(error):
            raise

        torch.cuda.empty_cache()
        result = {
            "status": "out_of_memory",
            "batch_size": batch_size,
            "steps_requested": steps,
            "error_type": type(error).__name__,
            "max_memory_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024 ** 3
            ),
            "max_memory_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024 ** 3
            ),
        }
        save_json(result, result_path)
        print(f"Batch probe result: {result}")
        return

    step_seconds = timing_callback.median_step_seconds()
    result = {
        "status": "success",
        "batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "steps_completed": trainer.state.global_step,
        "median_step_seconds_after_first_step": step_seconds,
        "sequences_per_second": (
            batch_size / step_seconds if step_seconds else None
        ),
        "token_positions_per_second": (
            batch_size * MAX_LENGTH / step_seconds
            if step_seconds
            else None
        ),
        "max_memory_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024 ** 3
        ),
        "max_memory_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024 ** 3
        ),
        "torch_compile": True,
        "bf16": True,
        "tf32": True,
    }
    save_json(result, result_path)
    print(f"Batch probe result: {result}")


def parse_args():
    parser = argparse.ArgumentParser(description="HW1 diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    untrained_parser = subparsers.add_parser(
        "evaluate-untrained",
        help="Evaluate the untrained model",
    )
    untrained_parser.add_argument(
        "--results-dir",
        default="./results",
        help="Results directory",
    )

    probe_parser = subparsers.add_parser(
        "probe-batch",
        help="Test one batch size",
    )
    probe_parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Per-device batch size",
    )
    probe_parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of optimizer steps",
    )
    probe_parser.add_argument(
        "--results-dir",
        default="./results",
        help="Results directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "evaluate-untrained":
        evaluate_untrained(results_dir=args.results_dir)
    elif args.command == "probe-batch":
        probe_batch_size(
            batch_size=args.batch_size,
            steps=args.steps,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
