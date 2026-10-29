"""Plot saved train/test experiment results without loading or rerunning a model."""

import json
from pathlib import Path


def plot_experiment(experiment_dir, *, save=True):
    """Return named Matplotlib figures from the train/test experiment folder.
    
    Uses frozen artifacts only; no threshold fitting occurs here. With save=True,
    writes PNG and SVG to <experiment_dir>/plots. Call plt.show() in a notebook.
    
    Args:
        experiment_dir: Path to a saved experiment folder containing the predictions or train/test artifacts required by this operation.
        save: Whether to write generated plots or analysis artifacts to disk (default True).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(experiment_dir)
    train = json.loads((root / "train/threshold_analysis/accuracy_threshold.json").read_text())
    report = json.loads((root / "test/threshold_comparison.json").read_text())
    rows = [json.loads(line) for line in (root / "test/threshold_comparison.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No successful test predictions to plot")
    threshold = report["threshold"]
    styles = {"native": ("Native model", "#64748b"),
              "threshold_0.5": ("Threshold 0.5", "#d97706"),
              "train_threshold": ("Train-selected threshold", "#2563eb")}
    figures = {}

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 110, "savefig.dpi": 180}):
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        curve = sorted(train["curve"], key=lambda p: p["threshold"])
        ax.step([p["threshold"] for p in curve], [p["accuracy"] for p in curve],
                where="pre", color="#2563eb", label="Training accuracy")
        ax.axvline(.5, color="#d97706", linestyle="--", label="Default threshold 0.5")
        ax.axvline(threshold, color="#16a34a", linestyle=":", label=f"Frozen threshold {threshold:.5g}")
        ax.scatter([threshold], [train["best"]["accuracy"]], color="#16a34a", zorder=5)
        ax.set(xlim=(0, max(1, threshold)), ylim=(0, 1.04), xlabel="P(positive) decision threshold",
               ylabel="Accuracy", title="Threshold selection uses training labels only")
        ax.legend(loc="best")
        ax.grid(alpha=.2)
        figures["training_threshold"] = fig

        fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        keys = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(keys))
        width = .24
        for index, (name, (label, color)) in enumerate(styles.items()):
            scores = report["comparison"][name]
            values = [scores.get(k) if scores.get(k) is not None else np.nan for k in keys]
            bars = ax.bar(x + (index - 1) * width, values, width, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.1%}" if np.isfinite(v) else "N/A" for v in values], fontsize=8, padding=3)
        n = report["comparison"]["native"]["scored"]
        attempted = report["comparison"]["native"]["attempted"]
        ax.set(xticks=x, xticklabels=["Accuracy", "Precision", "Recall", "F1"], ylim=(0, 1.17),
               ylabel="Score", title=f"Test performance — same {n}/{attempted} scored examples for all methods")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3)
        ax.grid(axis="y", alpha=.2)
        figures["test_metrics"] = fig

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
        maximum = max(report["comparison"][name][key] for name in styles for key in ("tn", "fp", "fn", "tp"))
        for ax, (name, (label, _)) in zip(axes, styles.items()):
            scores = report["comparison"][name]
            counts = np.array([[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]])
            image = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, maximum))
            for (y, x), value in np.ndenumerate(counts):
                ax.text(x, y, str(value), ha="center", va="center", fontsize=16,
                        color="white" if value > maximum / 2 else "#172554")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
                   yticklabels=["Negative", "Positive"], xlabel="Predicted class", ylabel="Actual class",
                   title=f"{label}\nAccuracy {scores['accuracy']:.1%}; errors {scores['errors']}")
        fig.colorbar(image, ax=axes, shrink=.7, label="Examples (shared scale)")
        fig.suptitle("Test confusion matrices")
        figures["test_confusion_matrices"] = fig

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        ax = axes[0]
        for label, name, color in [(0, "Actual negative", "#64748b"), (1, "Actual positive", "#2563eb")]:
            values = [r["p_positive"] for r in rows if r["label"] == label]
            ax.hist(values, bins=np.linspace(0, 1, 21), histtype="stepfilled", alpha=.4, label=name, color=color)
        ax.axvline(.5, color="#d97706", linestyle="--", label="Threshold 0.5")
        ax.axvline(threshold, color="#16a34a", linestyle=":", label=f"Train threshold {threshold:.5g}")
        ax.set(xlabel="P(positive)", ylabel="Test examples", title="Test score distributions", xlim=(0, max(1, threshold)))
        ax.legend(fontsize=8)
        ax = axes[1]
        bars = ax.bar(["Fixed", "Broken"], [report["fixed"], report["broken"]], color=["#16a34a", "#dc2626"])
        ax.bar_label(bars, padding=4)
        ax.set(ylabel="Test examples", title=f"Train threshold vs native predictions\nAccuracy change: {report['accuracy_gain_percentage_points']:+.2f} percentage points",
               ylim=(0, max(1, report["fixed"], report["broken"]) * 1.25))
        ax.grid(axis="y", alpha=.2)
        figures["test_scores_and_changes"] = fig

    if save:
        directory = root / "plots"
        directory.mkdir(exist_ok=True)
        for name, figure in figures.items():
            for extension in ("png", "svg"):
                figure.savefig(directory / f"{name}.{extension}", bbox_inches="tight")
    return figures


def plot_multiclass_results(experiment_dir, *, save=True):
    """Display native multiclass quality separately from binary threshold tuning.
    
    Args:
        experiment_dir: Path to a saved experiment folder containing the predictions or train/test artifacts required by this operation.
        save: Whether to write generated plots or analysis artifacts to disk (default True).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(experiment_dir)
    report = json.loads((root / "test/multiclass_metrics.json").read_text())
    classes = report["classes"]
    matrix = np.array(report["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for (y, x), count in np.ndenumerate(matrix):
        ax.text(x, y, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "#172554", fontsize=13)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted class", ylabel="Actual class", title="Native multiclass confusion matrix")
    fig.colorbar(image, ax=ax, shrink=.7, label="Examples")
    ax = axes[1]
    positions = np.arange(len(classes))
    for index, (metric, color) in enumerate((("precision", "#64748b"), ("recall", "#d97706"), ("f1", "#2563eb"))):
        bars = ax.bar(positions + (index - 1) * .24,
                      [report["per_class"][name][metric] for name in classes], .24, label=metric, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set(xticks=positions, xticklabels=classes, ylim=(0, 1.14), ylabel="Score", title="Per-class test performance")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3)
    fig.suptitle(f"Native multiclass accuracy {report['accuracy']:.1%}; macro-F1 {report['macro_f1']:.3f} — no class threshold tuning")
    if save:
        output = root / "plots"
        output.mkdir(exist_ok=True)
        for extension in ("png", "svg"):
            fig.savefig(output / f"native_multiclass.{extension}", dpi=180, bbox_inches="tight")
    return fig
