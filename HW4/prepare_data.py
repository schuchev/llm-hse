import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from experiment_configs import (
    DATA_DIR,
    DATASET_NAME,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_TEST,
    NUM_TRAIN,
    NUM_VALIDATION,
    SEED,
    SYSTEM_PROMPT,
)


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def make_example(row, index):
    context = clean_text(row["context"])
    question = clean_text(row["question"])
    answer = clean_text(row["answer"])
    return {
        "id": str(row.get("id", index)),
        "context": context,
        "question": question,
        "answer": answer,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос:\n{question}",
            },
        ],
        "completion": [{"role": "assistant", "content": answer}],
    }


def fits_in_max_length(example, tokenizer):
    messages = [*example["prompt"], *example["completion"]]
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(token_ids) <= MAX_LENGTH


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values, fraction):
    """Простая квантиль для отчёта о длинах последовательностей."""
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def main():
    total = NUM_TRAIN + NUM_VALIDATION + NUM_TEST
    print(f"Downloading {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME, split="train")
    print(f"Source rows: {len(raw):,}; columns: {raw.column_names}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    required_columns = {"context", "question", "answer"}
    missing = required_columns - set(raw.column_names)
    if missing:
        raise ValueError(f"Dataset misses expected columns: {sorted(missing)}")

    valid_indices = [
        index
        for index, row in enumerate(raw)
        if clean_text(row["context"]) and clean_text(row["question"]) and clean_text(row["answer"])
    ]
    if len(valid_indices) < total:
        raise ValueError(f"Only {len(valid_indices)} valid rows, but {total} are required")

    random.Random(SEED).shuffle(valid_indices)
    rows = []
    too_long = 0
    for index in valid_indices:
        example = make_example(raw[index], index)
        if fits_in_max_length(example, tokenizer):
            rows.append(example)
        else:
            too_long += 1
        if len(rows) == total:
            break
    if len(rows) < total:
        raise ValueError(f"Only {len(rows)} examples fit in MAX_LENGTH={MAX_LENGTH}")
    train_end = NUM_TRAIN
    validation_end = NUM_TRAIN + NUM_VALIDATION
    splits = {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }

    DATA_DIR.mkdir(exist_ok=True)
    for name, split_rows in splits.items():
        save_jsonl(DATA_DIR / f"{name}.jsonl", split_rows)

    manual_ids = [row["id"] for row in random.Random(SEED).sample(splits["test"], 8)]
    (DATA_DIR / "manual_example_ids.json").write_text(
        json.dumps(manual_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    answer_lengths = [
        len(tokenizer(row["answer"], add_special_tokens=False)["input_ids"])
        for row in rows
    ]
    metadata = {
        "source_dataset": DATASET_NAME,
        "seed": SEED,
        "max_length_tokens": MAX_LENGTH,
        "too_long_examples_seen": too_long,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "answer_length_tokens": {
            "median": percentile(answer_lengths, 0.5),
            "p95": percentile(answer_lengths, 0.95),
            "max": max(answer_lengths),
        },
    }
    (DATA_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nExample:")
    print(json.dumps(splits["train"][0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
