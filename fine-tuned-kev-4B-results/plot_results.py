import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
TRAINING = ROOT / "mine-v2" / "training_metrics.json"
MINE_REPORT = ROOT / "mine-v2-eval-3" / "report.json"
BASELINE_REPORT = ROOT / "baseline-eval-3" / "report.json"
OUTPUT = ROOT / "training_vs_baseline.png"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def rolling_median(values: list[float], window: int = 31) -> np.ndarray:
    return np.array(
        [
            np.median(values[max(0, index - window + 1) : index + 1])
            for index in range(len(values))
        ]
    )


def main() -> None:
    training = load_json(TRAINING)
    mine_report = load_json(MINE_REPORT)
    baseline_report = load_json(BASELINE_REPORT)
    mine = mine_report["clean"]
    baseline = baseline_report["clean"]

    colors = {"Fine-tuned": "#2563eb", "Baseline": "#f97316"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(bottom=0.14, top=0.93, hspace=0.35)
    fig.suptitle(
        "Fine-tuned KEV model vs baseline",
        fontsize=18,
        fontweight="bold",
    )

    steps = np.arange(1, len(training["step_seconds"]) + 1)
    axes[0, 0].plot(
        steps,
        training["step_seconds"],
        color="#94a3b8",
        linewidth=0.8,
        alpha=0.6,
        label="Per-step",
    )
    axes[0, 0].plot(
        steps,
        rolling_median(training["step_seconds"]),
        color="#2563eb",
        linewidth=2,
        label="31-step median",
    )
    axes[0, 0].set_title("Training step time")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    grad = training["grad_norm"]
    epochs = [item["epoch"] + 1 for item in grad]
    means = [item["mean"] for item in grad]
    maxima = [item["max"] for item in grad]
    width = 0.35
    positions = np.arange(len(epochs))
    axes[0, 1].bar(
        positions - width / 2,
        means,
        width,
        label="Mean",
        color="#2563eb",
    )
    axes[0, 1].bar(
        positions + width / 2,
        maxima,
        width,
        label="Max",
        color="#93c5fd",
    )
    axes[0, 1].set_xticks(positions, [f"Epoch {epoch}" for epoch in epochs])
    axes[0, 1].set_title("Gradient norm")
    axes[0, 1].set_ylabel("Norm")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.05,
        0.9,
        "Training summary",
        fontsize=14,
        fontweight="bold",
        transform=axes[0, 2].transAxes,
    )
    eval_summary = (
        "\nEvaluation summary\n"
        f"Accuracy: {mine['acc']:.4f} vs {baseline['acc']:.4f}\n"
        f"ECE: {mine['ece']:.4f} vs {baseline['ece']:.4f}\n"
        f"Coverage @ 0.9: {mine['coverage_at_0_9']:.1%} vs "
        f"{baseline['coverage_at_0_9']:.1%}\n"
        f"Median latency: {mine_report['latency_ms']['median']:.1f} vs "
        f"{baseline_report['latency_ms']['median']:.1f} ms"
    )
    axes[0, 2].text(
        0.05,
        0.32,
        eval_summary,
        fontsize=11,
        linespacing=1.6,
        verticalalignment="top",
        transform=axes[0, 2].transAxes,
    )
    summary = (
        f"Records: {training['records_seen']:,}\n"
        f"Optimizer steps: {training['optimizer_steps']:,}\n"
        f"Forward tokens: {training['forward_tokens']:,}\n"
        f"Wall time: {training['wall_seconds'] / 3600:.2f} h\n"
        f"Peak GPU memory: {training['peak_device_bytes'] / 2**30:.2f} GiB\n"
        f"Clipped steps: {sum(item['clipped_steps'] for item in grad):,}"
    )
    axes[0, 2].text(
        0.05,
        0.78,
        summary,
        fontsize=12,
        linespacing=1.7,
        verticalalignment="top",
        transform=axes[0, 2].transAxes,
    )

    metric_specs = [
        ("Accuracy", "acc", (0.75, 1.01), True),
        ("NLL (lower is better)", "nll", None, False),
        ("Brier score (lower is better)", "brier", None, False),
    ]
    for axis, (title, key, limits, _) in zip(axes[1], metric_specs):
        labels = list(colors)
        values = [mine[key], baseline[key]]
        bars = axis.bar(labels, values, color=[colors[label] for label in labels])
        axis.set_title(title)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Score")
    axes[1, 2].set_ylabel("Score")

    fig.text(
        0.5,
        0.025,
        (
            f"Evaluation set: n={mine['n']} | "
            f"Fine-tuned accuracy +{mine['acc'] - baseline['acc']:.1%}; "
            f"NLL {mine['nll'] / baseline['nll']:.1%} of baseline"
        ),
        ha="center",
            fontsize=10,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
