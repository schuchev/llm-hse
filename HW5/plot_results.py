"""Create report figures from saved GRPO logs and BFCL metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
SMOOTHING_WINDOW = 10
EXPERIMENTS = [
    "rl_exact",
    "rl_shaped",
    "rl_shaped_g8",
    "rl_shaped_temp1",
    "rl_shaped_temp1_multicall",
]


def moving_average(values: list[float], window: int = SMOOTHING_WINDOW) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / min(index + 1, window)
        for index in range(len(values))
    ]


def train_series(logs: list[dict], key: str) -> tuple[list[int], list[float]]:
    rows = [row for row in logs if "step" in row and key in row]
    return [int(row["step"]) for row in rows], [float(row[key]) for row in rows]


def draw_series(axis, steps: list[int], values: list[float], label: str, color=None) -> None:
    if not steps:
        return
    line = axis.plot(steps, moving_average(values), label=label, color=color, linewidth=2)[0]
    axis.scatter(steps, values, color=line.get_color(), alpha=0.16, s=10)


def plot_training_run(name: str) -> None:
    log_path = RESULTS_DIR / name / "log_history.json"
    if not log_path.exists():
        print(f"Skipping {name}: {log_path} not found")
        return

    logs = json.loads(log_path.read_text())
    figure, (loss_axis, reward_axis) = plt.subplots(1, 2, figsize=(13, 4.6))

    loss_steps, losses = train_series(logs, "loss")
    draw_series(loss_axis, loss_steps, losses, "train loss")
    loss_axis.set(title="Train loss", xlabel="Optimizer step", ylabel="Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    reward_steps, rewards = train_series(logs, "reward")
    draw_series(reward_axis, reward_steps, rewards, "total reward")

    component_keys = [
        ("rewards/exact_match_reward/mean", "exact reward"),
        ("rewards/format_reward/mean", "format component"),
        ("rewards/function_reward/mean", "function component"),
        ("rewards/argument_reward/mean", "argument component"),
    ]
    for key, label in component_keys:
        steps, values = train_series(logs, key)
        draw_series(reward_axis, steps, values, label)

    eval_steps, eval_rewards = train_series(logs, "eval_reward")
    if eval_steps:
        reward_axis.scatter(
            eval_steps,
            eval_rewards,
            label="validation reward",
            color="black",
            marker="x",
            s=50,
            zorder=3,
        )

    reward_axis.set(title="Reward", xlabel="Optimizer step", ylabel="Reward")
    reward_axis.grid(alpha=0.25)
    reward_axis.legend(fontsize=8)

    figure.suptitle(f"{name} (moving average: {SMOOTHING_WINDOW} steps)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_bfcl_overall() -> None:
    scores: dict[str, float] = {}
    paths = {"base": RESULTS_DIR / "base" / "metrics.json"}
    paths.update({name: RESULTS_DIR / name / "bfcl" / "metrics.json" for name in EXPERIMENTS})

    for name, path in paths.items():
        if path.exists():
            scores[name] = json.loads(path.read_text())["overall"]

    if not scores:
        return

    figure, axis = plt.subplots(figsize=(10, 4.5))
    names, values = list(scores), list(scores.values())
    colors = ["#6c757d" if name == "base" else "#247ba0" for name in names]
    bars = axis.bar(names, values, color=colors)
    axis.set(title="BFCL overall", ylabel="Macro accuracy", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "bfcl_overall.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        plot_training_run(name)
    plot_bfcl_overall()
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
