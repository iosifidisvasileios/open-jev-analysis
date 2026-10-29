"""Compare saved train/test experiment folders without loading models."""

import csv
import hashlib
import json
import math
from pathlib import Path

from .core import probability
from .experiments import _classification_metrics


def _load_split(root, split):
    """Load split.
    
    Args:
        root: Experiment root directory path.
        split: Split name recorded in the manifest, normally train, validation, or test.
    """
    folder = root / split
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("split") != split or manifest.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError(f"{folder}: expected completed {split} predictions")
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if not rows or len(rows) != manifest.get("examples"):
        raise ValueError(f"{folder}: prediction count differs from manifest")
    identity = []
    for row in rows:
        if not isinstance(row.get("text"), str) or type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"{folder}: invalid source text/label")
        identity.append((row["text"].strip(), row["label"]))
        if row.get("error") is None:
            probability(row["p_positive"])
            if type(row.get("predicted_label")) is not int or row["predicted_label"] not in (0, 1):
                raise ValueError(f"{folder}: invalid native prediction")
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return rows, manifest, identity, digest


def compare_experiments(experiment_dirs, *, names=None, include_tuned=True, native_only=False, output_dir=None):
    """Validate saved folders and compare binary accuracy on common test rows.
    
    Requires identical ordered test text/labels. With include_tuned=True, also
    requires identical ordered training text/labels and verifies each frozen
    threshold against its training predictions. Each model keeps its own threshold.
    Supports pre-split-manifest runs by checking actual saved rows, not folder names.
    Error rows are excluded from every model's metrics using the same intersection.
    The report retains per-model failures/coverage and training coverage separately.
    No model calls, fitting, or original-result mutations occur.
    
    Args:
        experiment_dirs: At least two distinct experiment-root paths, each containing test predictions and, when include_tuned=True, matching training and threshold artifacts.
        names: Optional unique display names aligned with experiment_dirs; None uses directory names.
        include_tuned: Include each model's frozen training threshold (default True); requires identical training selections and verifies artifact hashes.
        native_only: If True, report only native decisions; requires include_tuned=False. Default False also reports the fixed 0.5 rule.
        output_dir: Destination folder for downloaded data or saved reports, depending on the operation.
    """
    roots = [Path(p).resolve() for p in experiment_dirs]
    if native_only and include_tuned:
        raise ValueError("native_only requires include_tuned=False")
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("Provide at least two distinct experiment folders")
    names = list(names) if names is not None else [p.name for p in roots]
    if len(names) != len(roots) or len(set(names)) != len(names):
        raise ValueError("Provide one unique name for each experiment")
    loaded, warnings = [], []
    for name, root in zip(names, roots):
        test_rows, test_manifest, test_identity, test_hash = _load_split(root, "test")
        entry = {"name": name, "root": root, "test": test_rows, "manifest": test_manifest,
                 "test_identity": test_identity, "test_hash": test_hash}
        if include_tuned:
            train_rows, train_manifest, train_identity, train_hash = _load_split(root, "train")
            artifact = json.loads((root / "train/frozen_threshold.json").read_text())
            threshold = artifact["threshold"]
            if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1., math.inf):
                raise ValueError(f"{root}: invalid frozen threshold")
            if artifact.get("selection_split") != "train":
                raise ValueError(f"{root}: threshold was not selected on train")
            actual_hash = hashlib.sha256((root / "train/predictions.jsonl").read_bytes()).hexdigest()
            if artifact.get("training_predictions_sha256") != actual_hash:
                raise ValueError(f"{root}: frozen threshold does not match saved training predictions")
            if train_manifest["metadata"] != test_manifest["metadata"] or artifact["metadata"] != train_manifest["metadata"]:
                raise ValueError(f"{root}: train/test/threshold metadata differs")
            if {text for text, _ in train_identity} & {text for text, _ in test_identity}:
                raise ValueError(f"{root}: overlapping train/test texts")
            entry.update(train=train_rows, train_identity=train_identity, train_hash=train_hash, threshold=threshold)
        if loaded:
            if test_identity != loaded[0]["test_identity"]:
                raise ValueError(f"Test examples, labels, or order differ: {name} versus {names[0]}. Use the same dataset/profile.")
            if include_tuned and entry["train_identity"] != loaded[0]["train_identity"]:
                raise ValueError(f"Training examples, labels, or order differ: {name} versus {names[0]}.")
            if test_manifest["metadata"].get("question") != loaded[0]["manifest"]["metadata"].get("question"):
                warnings.append(f"{name}: question differs from {names[0]}; rubric effects may confound model comparisons.")
        loaded.append(entry)
    common = [i for i in range(len(loaded[0]["test"])) if all(e["test"][i].get("error") is None for e in loaded)]
    if not common:
        raise ValueError("No test examples successfully scored by every model")
    summary, native_vectors, descriptions = [], {}, []
    for entry in loaded:
        rows = [entry["test"][i] for i in common]
        native = [r["predicted_label"] for r in rows]
        native_vectors[entry["name"]] = native
        rules = {"native": native}
        if not native_only:
            rules["threshold_0.5"] = [int(r["p_positive"] >= .5) for r in rows]
        if include_tuned:
            rules["train_threshold"] = [int(r["p_positive"] >= entry["threshold"]) for r in rows]
        successful = sum(r.get("error") is None for r in entry["test"])
        training_errors = sum(r.get("error") is not None for r in entry.get("train", []))
        for method, decisions in rules.items():
            scores = _classification_metrics(rows, decisions)
            # Coverage/errors below describe the original runs, not the already
            # filtered common rows passed to the metrics helper.
            for key in ("coverage", "backend_errors", "attempted"):
                scores.pop(key, None)
            summary.append({"model": entry["name"], "method": method,
                            "threshold": entry.get("threshold") if method == "train_threshold" else (.5 if method == "threshold_0.5" else None),
                            **scores, "total_test_examples": len(entry["test"]),
                            "model_successful_test_examples": successful,
                            "model_test_backend_errors": len(entry["test"]) - successful,
                            "model_test_coverage": successful / len(entry["test"]),
                            "common_test_examples": len(common), "common_test_coverage": len(common) / len(entry["test"]),
                            "training_backend_errors": training_errors if include_tuned else None})
        descriptions.append({"name": entry["name"], "folder": str(entry["root"]),
                             "model_metadata": entry["manifest"]["metadata"],
                             "test_selection_sha256": entry["test_hash"],
                             "train_selection_sha256": entry.get("train_hash")})
        if training_errors:
            warnings.append(f"{entry['name']}: {training_errors} training inference failures; threshold fitted only on that model's successful training rows.")
    # Binary native agreement on the common test rows, independent of correctness.
    agreement = {a: {b: sum(x == y for x, y in zip(native_vectors[a], native_vectors[b])) / len(common)
                     for b in names} for a in names}
    report = {"experiments": descriptions, "test_selection_verified": True,
              "training_selection_verified": include_tuned,
              "common_test_row_indices": common, "summary": summary,
              "native_prediction_agreement": agreement, "warnings": warnings,
              "note": "Metrics use the identical intersection of successful test rows. Per-model and common coverage are reported separately. Every tuned threshold is selected on its own model's training scores; no test optimization occurs."}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return report


def plot_model_comparison(report):
    """Return a Matplotlib accuracy chart from compare_experiments().
    
    Args:
        report: Saved comparison report returned by compare_experiments; contains summary rows and common test membership.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = [e["name"] for e in report["experiments"]]
    methods = list(dict.fromkeys(r["method"] for r in report["summary"]))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2.3), 4.6), layout="constrained")
    width = .8 / len(methods)
    lookup = {(r["model"], r["method"]): r["accuracy"] for r in report["summary"]}
    for index, method in enumerate(methods):
        bars = ax.bar(np.arange(len(names)) + (index - (len(methods) - 1) / 2) * width,
                      [lookup[(name, method)] for name in names], width, label=method)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(xticks=np.arange(len(names)), xticklabels=names, ylim=(0, 1.13), ylabel="Test accuracy",
           title=f"Model comparison on {len(report['common_test_row_indices'])} shared successful test examples")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=len(methods))
    return fig
