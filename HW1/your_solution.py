import argparse
import json
import math
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    set_seed,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import torch
import time

from experiment_configs import (
    build_training_config,
    DEFAULT_BASE_BATCH_SIZE,
    EXPERIMENT_ORDER,
    get_attention_implementation,
    get_experiment_metadata,
    SEED,
)


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

GENERATION_PROMPTS = [
    "Искусственный интеллект — это область информатики, которая",
    "Москва — столица России. Город расположен",
    "Однажды зимним вечером маленький мальчик",
]


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """
    Prepare the tokenizer for fixed-length causal language modeling.
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    Tokenize a batch of texts for causal language modeling.
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    labels = [
        [token_id if mask == 1 else -100 for token_id, mask in zip(input_ids, attention_mask)]
        for input_ids, attention_mask in zip(
            tokenized[INPUT_IDS],
            tokenized[ATTENTION_MASK],
        )
    ]

    return {
        LABELS: labels,
        INPUT_IDS: tokenized[INPUT_IDS],
        ATTENTION_MASK: tokenized[ATTENTION_MASK],
    }


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    Save a dataset as sequential parquet shards.
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)

    for index in range(num_shards):
        shard = ds.shard(
            num_shards=num_shards,
            index=index,
            contiguous=True,
        )
        output_path = os.path.join(output_dir, f"{index:05d}.parquet")
        shard.to_parquet(output_path)


def prepare_dataset():
    """
    Download, tokenize, and save the Wikipedia dataset.
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        desc="Tokenizing Wikipedia",
    )

    save_as_parquets(tokenized_dataset)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    Load the tokenized dataset from sorted parquet shards.
    - List only parquet files in data_dir, sorted by filename
    - Load them using load_dataset('parquet', data_files=...)
    - Return the 'train' split
    """
    parquet_files = sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".parquet")
        and os.path.isfile(os.path.join(data_dir, filename))
    )

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dataset = load_dataset("parquet", data_files=parquet_files)
    return dataset["train"]


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(
    tokenizer,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Attention implementation: {attn_implementation}")
    
    return model


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def generate_samples(model, tokenizer):
    model.eval()
    samples = []

    for prompt in GENERATION_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }
        prompt_length = inputs[INPUT_IDS].shape[1]

        with torch.no_grad():
            greedy_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            set_seed(SEED)
            sampled_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        samples.append(
            {
                "prompt": prompt,
                "greedy": tokenizer.decode(
                    greedy_ids[0, prompt_length:],
                    skip_special_tokens=True,
                ),
                "sampled": tokenizer.decode(
                    sampled_ids[0, prompt_length:],
                    skip_special_tokens=True,
                ),
            }
        )

    return {
        "seed": SEED,
        "max_new_tokens": 64,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "samples": samples,
    }


def train_model(training_config):
    """
    Run the complete training and evaluation pipeline.
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from the selected experiment configuration
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Save metric history to trainer_state.json for local loss plots
    """
    experiment_name = training_config["run_name"]
    experiment_dir = training_config["output_dir"]

    if os.path.isdir(experiment_dir) and os.listdir(experiment_dir):
        raise FileExistsError(
            f"Experiment directory is not empty: {experiment_dir}"
        )

    os.makedirs(experiment_dir, exist_ok=True)
    experiment_metadata = get_experiment_metadata(
        experiment_name,
        training_config,
    )
    save_json(
        {
            "experiment": experiment_metadata,
            "timeout_seconds": MAX_TRAINING_TIME_SECONDS,
            "training_args": training_config,
        },
        os.path.join(experiment_dir, "config.json"),
    )

    set_seed(SEED)
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    model_dtype = (
        torch.bfloat16
        if training_config["bf16"]
        else torch.float32
    )
    attn_implementation = get_attention_implementation(
        experiment_name,
        training_config,
    )
    model = create_model(
        tokenizer,
        torch_dtype=model_dtype,
        attn_implementation=attn_implementation,
    )
    training_args = TrainingArguments(**training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)]  # dont change
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    if eval_loss is not None:
        try:
            eval_results["perplexity"] = math.exp(eval_loss)
        except OverflowError:
            eval_results["perplexity"] = None

    print(f"Final evaluation results: {eval_results}")
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    if eval_results.get("perplexity") is not None:
        trainer.log({"eval_perplexity": eval_results["perplexity"]})
    trainer.save_state()

    generations = generate_samples(trainer.model, tokenizer)
    save_json(
        generations,
        os.path.join(experiment_dir, "generations.json"),
    )

    effective_batch_size = (
        training_config["per_device_train_batch_size"]
        * training_config["gradient_accumulation_steps"]
    )
    summary = {
        "experiment_name": experiment_name,
        "experiment": experiment_metadata,
        "timeout_seconds": MAX_TRAINING_TIME_SECONDS,
        "global_step": trainer.state.global_step,
        "estimated_input_tokens": (
            trainer.state.global_step
            * effective_batch_size
            * MAX_LENGTH
        ),
        "best_eval_loss": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval_loss": eval_loss,
        "perplexity": eval_results.get("perplexity"),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_results,
    }
    save_json(summary, os.path.join(experiment_dir, "summary.json"))


def parse_args():
    parser = argparse.ArgumentParser(description="HW1 model pretraining")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare-data",
        help="Prepare the dataset",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model",
    )
    train_parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_ORDER,
        default="reference",
        help="Experiment name",
    )
    train_parser.add_argument(
        "--base-batch-size",
        type=int,
        default=DEFAULT_BASE_BATCH_SIZE,
        help="Base per-device batch size",
    )
    train_parser.add_argument(
        "--results-dir",
        default="./results",
        help="Results directory",
    )
    train_parser.add_argument(
        "--tracker",
        choices=("none", "tensorboard", "wandb"),
        default="tensorboard",
        help="Logging backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "prepare-data":
        prepare_dataset()
    elif args.command == "train":
        training_config = build_training_config(
            experiment_name=args.experiment,
            base_batch_size=args.base_batch_size,
            results_dir=args.results_dir,
            tracker=args.tracker,
        )
        train_model(training_config)


if __name__ == "__main__":
    main()
