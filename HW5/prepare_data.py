import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3.5-2B"
SEED = 42
VALIDATION_FRACTION = 0.1
OUTPUT_DIR = Path("data")
MAX_PROMPT_TOKENS = 2048
MAX_TARGET_TOKENS = 512


def split_top_level(text: str, separator: str = ",") -> list[str]:
    pieces, start, depth, quote, escaped = [], 0, 0, None, False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{" :
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def parse_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"\'')


def parse_toolace_calls(text: str) -> list[dict[str, Any]] | None:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None

    calls = []
    for call_text in split_top_level(text[1:-1]):
        open_paren = call_text.find("(")
        if open_paren < 1 or not call_text.endswith(")"):
            return None

        name = call_text[:open_paren].strip()
        arguments: dict[str, Any] = {}
        for argument in split_top_level(call_text[open_paren + 1 : -1]):
            key, separator, value = argument.partition("=")
            if not separator or not key.strip():
                return None
            arguments[key.strip()] = parse_value(value.strip())
        calls.append({"name": name, "arguments": arguments})
    return calls


def normalize_type(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_type(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: normalize_type(item) for key, item in value.items()}
    if isinstance(normalized.get("type"), str):
        normalized["type"] = {
            "dict": "object",
            "float": "number",
            "tuple": "array",
            "any": "string",
        }.get(normalized["type"], normalized["type"])
    return normalized


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if "name" in tool:
        return {"type": "function", "function": normalize_type(tool)}

    properties = {
        argument["parameter_name"]: normalize_type(
            {
                "type": argument.get("type", "string"),
                "description": argument.get("description", ""),
                **({"items": argument["items"]} if "items" in argument else {}),
            }
        )
        for argument in tool.get("arguments", [])
    }
    return {
        "type": "function",
        "function": {
            "name": tool["tool_name"],
            "description": tool.get("definition", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": tool.get("required") or [],
            },
        },
    }


def extract_tools(system_message: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(system_message):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(system_message[start:])
        except json.JSONDecodeError:
            continue
        tools = parsed if isinstance(parsed, list) else [parsed]
        if tools and all(
            isinstance(tool, dict) and ("name" in tool or "tool_name" in tool)
            for tool in tools
        ):
            return [normalize_tool(tool) for tool in tools]
    raise ValueError("ToolACE function specification was not found")


def convert_dialog(dialog_id: int, row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    tools = extract_tools(row["system"])
    messages = [{"role": "system", "content": "You are a helpful function-calling assistant."}]
    examples = []
    skipped_calls = 0
    previous_source_role: str | None = None

    for turn in row["conversations"]:
        role = {"user": "user", "assistant": "assistant", "tool": "tool"}[turn["from"]]
        content = turn["value"]

        if role == "assistant":
            calls = parse_toolace_calls(content)
            if calls:
                examples.append(
                    {
                        "dialog_id": dialog_id,
                        "messages": list(messages),
                        "tools": tools,
                        "target_calls": calls,
                        "target_content": "",
                        "target_type": "tool_call",
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": call} for call in calls
                        ],
                    }
                )
            elif calls == [] or (
                previous_source_role == "user"
                and not (content.strip().startswith("[") and content.strip().endswith("]"))
            ):
                examples.append(
                    {
                        "dialog_id": dialog_id,
                        "messages": list(messages),
                        "tools": tools,
                        "target_calls": [],
                        "target_content": content,
                        "target_type": "no_call",
                    }
                )
                messages.append({"role": "assistant", "content": content})
            else:
                skipped_calls += content.strip().startswith("[")
                messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": role, "content": content})
        previous_source_role = turn["from"]

    return examples, skipped_calls


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_lengths(example: dict[str, Any], tokenizer: Any) -> tuple[int, int]:
    prompt_ids = tokenizer.apply_chat_template(
        example["messages"],
        tools=example["tools"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target_message = {
        "role": "assistant",
        "content": example["target_content"],
    }
    if example["target_calls"]:
        target_message["tool_calls"] = [
            {"type": "function", "function": call}
            for call in example["target_calls"]
        ]
    full_ids = tokenizer.apply_chat_template(
        [*example["messages"], target_message],
        tools=example["tools"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(prompt_ids), len(full_ids) - len(prompt_ids)


def main() -> None:
    dataset = load_dataset("Team-ACE/ToolACE", split="train")
    dialog_ids = list(range(len(dataset)))
    random.Random(SEED).shuffle(dialog_ids)
    validation_size = round(len(dialog_ids) * VALIDATION_FRACTION)
    validation_dialog_ids = set(dialog_ids[:validation_size])

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    turns_per_dialog = Counter()
    skipped_calls = 0
    dialogs_without_tools = 0
    bracketed_targets_without_tools = 0

    for dialog_id, row in enumerate(dataset):
        turns_per_dialog[len(row["conversations"])] += 1
        try:
            examples, skipped = convert_dialog(dialog_id, row)
        except ValueError:
            dialogs_without_tools += 1
            bracketed_targets_without_tools += sum(
                turn["from"] == "assistant"
                and turn["value"].strip().startswith("[")
                and turn["value"].strip().endswith("]")
                for turn in row["conversations"]
            )
            continue
        skipped_calls += skipped
        destination = validation_rows if dialog_id in validation_dialog_ids else train_rows
        destination.extend(examples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    filtered_by_prompt = 0
    filtered_by_target = 0

    def filter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal filtered_by_prompt, filtered_by_target
        kept = []
        for example in rows:
            prompt_length, target_length = token_lengths(example, tokenizer)
            if prompt_length > MAX_PROMPT_TOKENS:
                filtered_by_prompt += 1
                continue
            if target_length > MAX_TARGET_TOKENS:
                filtered_by_target += 1
                continue
            example["prompt_tokens"] = prompt_length
            example["target_tokens"] = target_length
            kept.append(example)
        return kept

    train_rows = filter_rows(train_rows)
    validation_rows = filter_rows(validation_rows)
    all_rows = train_rows + validation_rows
    prompt_lengths = [example["prompt_tokens"] for example in all_rows]
    target_lengths = [example["target_tokens"] for example in all_rows]
    calls_per_example = Counter(len(example["target_calls"]) for example in all_rows)
    tools_per_example = Counter(len(example["tools"]) for example in all_rows)
    called_functions = Counter(
        call["name"] for example in all_rows for call in example["target_calls"]
    )
    examples_by_target_type = Counter(example["target_type"] for example in all_rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_jsonl(OUTPUT_DIR / "toolace_train.jsonl", train_rows)
    save_jsonl(OUTPUT_DIR / "toolace_validation.jsonl", validation_rows)

    statistics = {
        "source_dialogs": len(dataset),
        "dialog_split": {"train": len(dialog_ids) - validation_size, "validation": validation_size},
        "examples": {"train": len(train_rows), "validation": len(validation_rows)},
        "examples_by_target_type": dict(sorted(examples_by_target_type.items())),
        "length_filter": {
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_target_tokens": MAX_TARGET_TOKENS,
            "filtered_by_prompt": filtered_by_prompt,
            "filtered_by_target": filtered_by_target,
        },
        "turns_per_dialog": dict(sorted(turns_per_dialog.items())),
        "calls_per_example": dict(sorted(calls_per_example.items())),
        "tools_per_example": dict(sorted(tools_per_example.items())),
        "most_frequent_target_functions": called_functions.most_common(20),
        "unparseable_bracketed_answers": skipped_calls,
        "dialogs_without_machine_readable_tools": dialogs_without_tools,
        "bracketed_targets_skipped_without_tools": bracketed_targets_without_tools,
        "prompt_tokens": {
            "median": sorted(prompt_lengths)[len(prompt_lengths) // 2],
            "p95": sorted(prompt_lengths)[int(len(prompt_lengths) * 0.95)],
            "max": max(prompt_lengths),
        },
        "target_tokens": {
            "median": sorted(target_lengths)[len(target_lengths) // 2],
            "p95": sorted(target_lengths)[int(len(target_lengths) * 0.95)],
            "max": max(target_lengths),
        },
    }
    (OUTPUT_DIR / "toolace_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
