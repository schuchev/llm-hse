import ast
import json
import re
from typing import Any


TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def parse_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text


def normalize_json_call(value: Any) -> list[dict[str, Any]] | None:
    values = value if isinstance(value, list) else [value]
    calls = []
    for item in values:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        calls.append({"name": name, "arguments": arguments})
    return calls


def parse_completion(text: str) -> tuple[list[dict[str, Any]], bool, bool]:
    text = text.strip()
    has_markup = any(tag in text for tag in ("<tool_call", "<function=", "<parameter="))

    if not has_markup:
        return [], False, True

    if (
        text.count("<tool_call>") != text.count("</tool_call>")
        or len(re.findall(r"<function=[^>]+>", text)) != text.count("</function>")
        or len(re.findall(r"<parameter=[^>]+>", text)) != text.count("</parameter>")
    ):
        return [], True, False

    blocks = TOOL_BLOCK_RE.findall(text)
    if not blocks or len(blocks) != text.count("<tool_call>"):
        return [], True, False

    calls: list[dict[str, Any]] = []
    for block in blocks:
        function_blocks = FUNCTION_RE.findall(block)
        if function_blocks:
            for name, function_content in function_blocks:
                arguments = {
                    parameter_name.strip(): parse_value(parameter_value)
                    for parameter_name, parameter_value in PARAMETER_RE.findall(
                        function_content
                    )
                }
                calls.append({"name": name.strip(), "arguments": arguments})
            continue

        try:
            json_calls = normalize_json_call(json.loads(block.strip()))
        except json.JSONDecodeError:
            json_calls = None
        if json_calls is None:
            return [], True, False
        calls.extend(json_calls)

    return calls, True, True


def values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            values_equal(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    return left == right


def argument_f1(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    if not predicted and not target:
        return 1.0
    correct = sum(
        key in predicted and values_equal(predicted[key], value)
        for key, value in target.items()
    )
    return 2 * correct / (len(predicted) + len(target))


def match_calls(
    predicted_calls: list[dict[str, Any]], target_calls: list[dict[str, Any]]
) -> list[float]:
    candidates = []
    for predicted_index, predicted in enumerate(predicted_calls):
        for target_index, target in enumerate(target_calls):
            if predicted["name"] == target["name"]:
                candidates.append(
                    (
                        argument_f1(predicted["arguments"], target["arguments"]),
                        predicted_index,
                        target_index,
                    )
                )

    used_predicted, used_target, scores = set(), set(), []
    for score, predicted_index, target_index in sorted(candidates, reverse=True):
        if predicted_index not in used_predicted and target_index not in used_target:
            used_predicted.add(predicted_index)
            used_target.add(target_index)
            scores.append(score)
    return scores


def score_completion(text: str, target_calls: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_calls, has_markup, valid_markup = parse_completion(text)

    if not target_calls:
        no_call = bool(text.strip()) and not predicted_calls and not has_markup
        return {
            "reward": float(no_call),
            "exact": no_call,
            "format_score": float(no_call),
            "function_f1": float(no_call),
            "argument_score": float(no_call),
            "predicted_calls": predicted_calls,
        }

    if not valid_markup or not predicted_calls:
        return {
            "reward": 0.0,
            "exact": False,
            "format_score": 0.0,
            "function_f1": 0.0,
            "argument_score": 0.0,
            "predicted_calls": predicted_calls,
        }

    matched_argument_scores = match_calls(predicted_calls, target_calls)
    matched_count = len(matched_argument_scores)
    function_f1 = 2 * matched_count / (len(predicted_calls) + len(target_calls))
    argument_score = sum(matched_argument_scores) / max(
        len(predicted_calls), len(target_calls)
    )
    reward = 0.10 + 0.25 * function_f1 + 0.65 * argument_score
    exact = len(predicted_calls) == len(target_calls) and argument_score == 1.0
    return {
        "reward": reward,
        "exact": exact,
        "format_score": 1.0,
        "function_f1": function_f1,
        "argument_score": argument_score,
        "predicted_calls": predicted_calls,
    }


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def tool_call_reward(
    completions: list[Any], target_calls: list[list[dict[str, Any]]], **_: Any
) -> list[float]:
    return [
        score_completion(completion_to_text(completion), target)["reward"]
        for completion, target in zip(completions, target_calls, strict=True)
    ]


def format_reward(
    completions: list[Any], target_calls: list[list[dict[str, Any]]], **_: Any
) -> list[float]:
    """The 0.10 weighted format component of the shaped reward."""
    return [
        0.10 * score_completion(completion_to_text(completion), target)["format_score"]
        for completion, target in zip(completions, target_calls, strict=True)
    ]


def function_reward(
    completions: list[Any], target_calls: list[list[dict[str, Any]]], **_: Any
) -> list[float]:
    """The 0.25 weighted function-set component of the shaped reward."""
    return [
        0.25 * score_completion(completion_to_text(completion), target)["function_f1"]
        for completion, target in zip(completions, target_calls, strict=True)
    ]


def argument_reward(
    completions: list[Any], target_calls: list[list[dict[str, Any]]], **_: Any
) -> list[float]:
    """The 0.65 weighted argument component of the shaped reward."""
    return [
        0.65 * score_completion(completion_to_text(completion), target)["argument_score"]
        for completion, target in zip(completions, target_calls, strict=True)
    ]


def exact_match_reward(
    completions: list[Any], target_calls: list[list[dict[str, Any]]], **_: Any
) -> list[float]:
    return [
        float(score_completion(completion_to_text(completion), target)["exact"])
        for completion, target in zip(completions, target_calls, strict=True)
    ]


def _run_self_tests() -> None:
    target = [{"name": "get_weather", "arguments": {"city": "Moscow", "days": 2}}]
    correct_json = '<tool_call>{"name":"get_weather","arguments":{"days":2,"city":"Moscow"}}</tool_call>'
    correct_xml = """<tool_call><function=get_weather>
<parameter=city>Moscow</parameter><parameter=days>2</parameter>
</function></tool_call>"""
    missing_argument = '<tool_call>{"name":"get_weather","arguments":{"city":"Moscow"}}</tool_call>'
    extra_call = correct_json + '<tool_call>{"name":"search","arguments":{}}</tool_call>'

    assert score_completion(correct_json, target)["reward"] == 1.0
    assert score_completion(correct_xml, target)["reward"] == 1.0
    assert 0 < score_completion(missing_argument, target)["reward"] < 1
    assert 0 < score_completion(extra_call, target)["reward"] < 1
    assert score_completion("The weather is nice.", target)["reward"] == 0.0
    assert score_completion("No suitable tool is available.", [])["reward"] == 1.0
    assert score_completion(correct_json, [])["reward"] == 0.0
    assert score_completion("<tool_call>{bad json}</tool_call>", target)["reward"] == 0.0
    print("Reward self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
