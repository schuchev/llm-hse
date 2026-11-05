import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_configs import EXPERIMENTS, MODEL_NAME, RESULTS_DIR


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def token_f1(prediction, reference):
    predicted_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def load_model(name):
    if name == "base":
        model_path = MODEL_NAME
        adapter_path = None
    else:
        model_path = str(RESULTS_DIR / name / "model")
        adapter_path = model_path if EXPERIMENTS[name]["method"] == "lora" else None
        if adapter_path:
            model_path = MODEL_NAME

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model.to("cuda").eval()


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts):
    inputs = tokenizer.apply_chat_template(
        prompts,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    ).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [
        text.strip()
        for text in tokenizer.batch_decode(
            output_ids[:, prompt_length:], skip_special_tokens=True
        )
    ]


def update_metrics(name, test_metrics):
    if name == "base":
        (RESULTS_DIR / "base_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    path = RESULTS_DIR / name / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["test_metrics"] = test_metrics
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def result_row(name, test_metrics):
    """Строка итоговой таблицы: качество плюс ресурсы, записанные train.py."""
    row = {"experiment": name, **test_metrics}
    if name == "base":
        return {"method": "base", **row}

    metrics = json.loads((RESULTS_DIR / name / "metrics.json").read_text(encoding="utf-8"))
    return {
        "method": metrics["method"],
        "train_seconds": metrics["train_seconds"],
        "global_step": metrics["global_step"],
        "trainable_parameters": metrics["trainable_parameters"],
        "trainable_percent": metrics["trainable_percent"],
        "peak_allocated_gib": metrics["peak_allocated_gib"],
        "peak_reserved_gib": metrics["peak_reserved_gib"],
        "validation_loss": metrics["eval_metrics"].get("eval_loss"),
        **row,
    }


def write_manual_comparison(names):
    """Собрать предсказания одних и тех же восьми вопросов в одну CSV-таблицу."""
    by_id = {}
    for name in names:
        path = RESULTS_DIR / name / "manual_examples.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            item = by_id.setdefault(
                row["id"],
                {"id": row["id"], "question": row["question"], "reference": row["reference"]},
            )
            item[name] = row["prediction"]

    rows = list(by_id.values())
    if not rows:
        return
    fields = ["id", "question", "reference", *names]
    with (RESULTS_DIR / "manual_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(name, rows, manual_ids):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_model(name)
    predictions = []
    batch_size = 32
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_predictions = generate_batch(
            model, tokenizer, [row["prompt"] for row in batch_rows]
        )
        for row, prediction in zip(batch_rows, batch_predictions):
            predictions.append({
                "id": row["id"],
                "question": row["question"],
                "reference": row["answer"],
                "prediction": prediction,
                "exact_match": float(normalize(prediction) == normalize(row["answer"])),
                "token_f1": token_f1(prediction, row["answer"]),
            })
        if len(predictions) % 48 == 0 or len(predictions) == len(rows):
            print(f"{name}: {len(predictions)}/{len(rows)}")

    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    test_metrics = {
        "exact_match": sum(row["exact_match"] for row in predictions) / len(predictions),
        "token_f1": sum(row["token_f1"] for row in predictions) / len(predictions),
        "num_test_examples": len(predictions),
    }
    manual_examples = [row for row in predictions if row["id"] in manual_ids]
    (output_dir / "manual_examples.json").write_text(
        json.dumps(manual_examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_metrics(name, test_metrics)
    del model
    torch.cuda.empty_cache()
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", choices=["base", *EXPERIMENTS])
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path("data/test.jsonl"))
    manual_ids = set(json.loads(Path("data/manual_example_ids.json").read_text(encoding="utf-8")))
    names = ["base", *EXPERIMENTS] if args.all else [args.experiment]
    table = []
    evaluated_names = []
    for name in names:
        if name != "base" and not (RESULTS_DIR / name / "model").exists():
            print(f"Skipping {name}: trained model was not found")
            continue
        test_metrics = evaluate(name, rows, manual_ids)
        row = result_row(name, test_metrics)
        table.append(row)
        evaluated_names.append(name)
        print(row)

    if args.all:
        fields = [
            "experiment", "method", "exact_match", "token_f1", "num_test_examples",
            "validation_loss", "train_seconds", "global_step", "trainable_parameters",
            "trainable_percent", "peak_allocated_gib", "peak_reserved_gib",
        ]
        with (RESULTS_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table)
        write_manual_comparison(evaluated_names)


if __name__ == "__main__":
    main()
