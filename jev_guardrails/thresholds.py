"""Find the empirical accuracy-maximizing binary decision threshold."""

import csv
import json
import math
from pathlib import Path


def optimize_accuracy_threshold(
    experiment_dir,
    *,
    score_column=None,
    label_column="label",
    baseline_threshold=0.5,
    save=True,
):
    """Read predictions.jsonl (or predictions.csv) and minimize FP + FN.
    
    Scores must be probabilities of label 1, NOT confidence in the predicted
    class. Auto-detects p_injection, p_unsafe, or p_positive. Supply score_column
    for other models. Predict 1 iff score >= threshold. Errors are excluded and
    counted. Labels are 0/1; no inference/model package is required.
    
    Evaluate every distinct decision boundary, grouping equal scores together.
    Ties prefer the candidate closest to baseline_threshold, then the lower one.
    A threshold just above 1 can represent predicting all-negative when p=1 is
    present. The selected accuracy is an in-sample optimum, not held-out quality.
    
    Args:
        experiment_dir: Path to a saved experiment folder containing the predictions or train/test artifacts required by this operation.
        score_column: Column containing P(binary label 1), not confidence in the selected class. None detects p_injection, p_unsafe, or p_positive.
        label_column: Ground-truth binary label column (default "label"), containing integer 0/1 values.
        baseline_threshold: Reference probability threshold (default 0.5); among equally accurate optima prefer the closest threshold, then the lower value.
        save: Whether to write generated plots or analysis artifacts to disk (default True).
    """
    folder = Path(experiment_dir)
    if not math.isfinite(baseline_threshold) or not 0 <= baseline_threshold <= 1:
        raise ValueError("baseline_threshold must be between 0 and 1")
    path = folder / "predictions.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        path = folder / "predictions.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    successful = [row for row in rows if row.get("error") in (None, "")]
    if not successful:
        raise ValueError("No successful predictions to optimize")
    if score_column is None:
        score_column = next((key for key in ("p_injection", "p_unsafe", "p_positive")
                             if all(key in row for row in successful)), None)
        if score_column is None:
            raise ValueError("Specify score_column containing P(label=1); confidence alone is insufficient")
    tasks = {row.get("task") for row in successful if row.get("task") is not None}
    if len(tasks) > 1:
        raise ValueError("Mixed tasks found; optimize one binary task at a time")
    points = []
    for index, row in enumerate(successful):
        score, label = row.get(score_column), row.get(label_column)
        if isinstance(score, bool) or isinstance(label, bool):
            raise ValueError(f"Row {index}: boolean score/label is not allowed")
        try:
            score, label = float(score), float(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {index}: missing or nonnumeric score/label") from exc
        if not math.isfinite(score) or not 0 <= score <= 1 or label not in (0, 1):
            raise ValueError(f"Row {index}: expected probability [0,1] and label 0/1")
        points.append((score, int(label)))
    n = len(points)
    positives = sum(label for _, label in points)
    negatives = n - positives

    def summary(threshold, tp, fp):
        """Summary.
        
        Args:
            threshold: Binary decision cutoff: predict positive when the positive probability is at least this value.
            tp: Input tp.
            fp: Input fp.
        """
        tn, fn = negatives - fp, positives - tp
        def ratio(a, b):
            """Ratio.
            
            Args:
                a: Numerator of the ratio.
                b: Denominator; zero produces an undefined metric instead of division.
            """
            return a / b if b else None
        return {
            "threshold": threshold, "accuracy": (tp + tn) / n,
            "errors": fp + fn, "correct": tp + tn,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, positives),
            "benign_block_rate": ratio(fp, negatives),
        }

    # O(n log n), with counts updated once per example, never splitting ties.
    points.sort(reverse=True)
    thresholds = sorted({score for score, _ in points} |
                        {0.0, 1.0, baseline_threshold, math.nextafter(points[0][0], math.inf)},
                        reverse=True)
    curve, cursor, tp, fp = [], 0, 0, 0
    for threshold in thresholds:
        while cursor < n and points[cursor][0] >= threshold:
            label = points[cursor][1]
            tp += label
            fp += 1 - label
            cursor += 1
        curve.append(summary(threshold, tp, fp))
    curve.sort(key=lambda item: item["threshold"])
    baseline = next(item for item in curve if item["threshold"] == baseline_threshold)
    best = min(curve, key=lambda item: (item["errors"],
                                      abs(item["threshold"] - baseline_threshold), item["threshold"]))
    native_rows = [row for row in successful if row.get("predicted_label") not in (None, "")]
    native_accuracy = None
    if len(native_rows) == n:
        # Native argmax can differ from >= 0.5 on exact ties.
        if any(float(row["predicted_label"]) not in (0, 1) for row in native_rows):
            raise ValueError("Native predicted_label must be 0 or 1")
        native_accuracy = sum(float(row["predicted_label"]) == float(row[label_column])
                              for row in native_rows) / n
    report = {
        "experiment_dir": str(folder.resolve()), "source_file": path.name,
        "score_column": score_column, "label_column": label_column,
        "decision_rule": "predict label 1 when score >= threshold",
        "attempted": len(rows), "scored": n, "excluded_errors": len(rows) - n,
        "coverage": n / len(rows), "positive_labels": positives, "negative_labels": negatives,
        "baseline": baseline, "best": best,
        "native_prediction_accuracy": native_accuracy,
        "accuracy_gain_percentage_points": 100 * (best["accuracy"] - baseline["accuracy"]),
        "errors_removed": baseline["errors"] - best["errors"],
        "equally_optimal_candidates": sum(item["errors"] == best["errors"] for item in curve),
        "warning": "Threshold selected and scored on the same data; accuracy is an in-sample optimum. Freeze it and evaluate on separate held-out data for generalization.",
        "curve": curve,
    }
    if save:
        output = folder / "threshold_analysis"
        output.mkdir(exist_ok=True)
        (output / "accuracy_threshold.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (output / "threshold_curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
    return report
