"""Reusable binary inference, train-only threshold fitting, and test comparison.

A model adapter is simply predict(text) -> {p_positive, predicted_label,
confidence, ...}. Dataset rows need a text column and an integer 0/1 label.
No model or Hugging Face dependencies are imported by this module.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from .core import probability
from .thresholds import optimize_accuracy_threshold


def _write_json(path, value):
    """Write json.
    
    Args:
        path: Filesystem path to the input or output file.
        value: Value to validate, hash, or serialize; probability() requires a finite numeric value in [0, 1].
    """
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _text_hash(text):
    """Text hash.
    
    Args:
        text: Input post as a string, without reference labels. Treated as untrusted evidence, not classifier instructions.
    """
    return hashlib.sha256(text.strip().encode()).hexdigest()


def _read_rows(folder):
    """Read rows.
    
    Args:
        folder: Directory containing predictions.jsonl.
    """
    return [json.loads(line) for line in (Path(folder) / "predictions.jsonl").read_text().splitlines() if line.strip()]


def assert_disjoint_splits(train, test, *, text_column="text"):
    """Reject exact text overlap after trimming surrounding whitespace.
    
    Args:
        train: Prepared training dataset; must not share trimmed text with test. DSPy selection expects id, text, label, and label_multiclass columns.
        test: Prepared held-out test dataset; membership is preserved. Text is used to detect leakage, not to tune prompts or thresholds.
        text_column: Dataset text column name (default "text"); trimmed text is used for overlap checks.
    """
    overlap = {_text_hash(row[text_column]) for row in train} & {_text_hash(row[text_column]) for row in test}
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping texts between train and test")


def make_choice_predictor(model, question, *, positive_class, negative_class, confidence_is_probability=True):
    """Native choice adapter; positive_class can be a key or collection of keys.
    
    With grouped positives, p_positive is their summed probability and the native
    predicted_label is the group containing the native argmax. Confidence remains
    the native selected-class probability, not the summed group probability.
    Set confidence_is_probability=False for APIs with a different native
    confidence definition; selected_class_probability is always saved separately.
    
    Args:
        model: Classifier instance for local inference, or model/deployment identifier when configuring a backend or SDK engine.
        question: Choice rubric dictionary with type, instructions, and criteria mapping class names to descriptions; copied before editing.
        positive_class: One class name or collection of names mapped to binary label 1; probabilities are summed for a collection.
        negative_class: Class name mapped to binary label 0, usually "normal".
        confidence_is_probability: Whether native confidence is a probability; False retains the server confidence separately from class probabilities.
    """
    question = deepcopy(question)
    positives = {positive_class} if isinstance(positive_class, str) else set(positive_class)
    if not positives or negative_class in positives or question.get("type") != "choice":
        raise ValueError("Expected disjoint positive/negative classes and a choice question")
    classes = {negative_class, *positives}
    if set(question.get("criteria", {})) != classes:
        raise ValueError("Question criteria must match the supplied class keys")

    def predict(text):
        """Predict.
        
        Args:
            text: Input post as a string, without reference labels. Treated as untrusted evidence, not classifier instructions.
        """
        answer = model.decide(text, deepcopy(question))
        if answer.get("type") != "choice" or set(answer.get("probabilities", {})) != classes:
            raise ValueError("Unexpected native answer type or class keys")
        probs = {k: probability(v) for k, v in answer["probabilities"].items()}
        if not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
            raise ValueError("Class probabilities do not sum to one")
        selected = answer["choice"]
        if selected not in probs or probs[selected] != max(probs.values()):
            raise ValueError("Native choice does not maximize class probability")
        confidence = probability(answer["confidence"])
        if confidence_is_probability and not math.isclose(confidence, probs[selected], abs_tol=1e-5):
            raise ValueError("Confidence differs from selected-class probability")
        return {"p_positive": min(1.0, sum(probs[key] for key in positives)), "predicted_label": int(selected in positives),
                "predicted_class": selected, "confidence": confidence,
                "selected_class_probability": probs[selected],
                "input_tokens": answer.get("input_tokens"), "raw_answer": answer}

    return predict


def make_client_choice_predictor(client, question, *, positive_class, negative_class,
                                 model_alias="open-jev", expected_model=None):
    """Adapt jev.client.Client.ask to the same text-only experiment interface.
    
    Args:
        client: Configured Open-Jev client exposing ask(); reused for classification requests.
        question: Choice rubric dictionary with type, instructions, and criteria mapping class names to descriptions; copied before editing.
        positive_class: One class name or collection of names mapped to binary label 1; probabilities are summed for a collection.
        negative_class: Class name mapped to binary label 0, usually "normal".
        model_alias: Model name sent to the server (default "open-jev"); this does not load a checkpoint.
        expected_model: Expected returned checkpoint identifier, usually from /health; None disables the identity check.
    """
    from urllib.error import HTTPError

    class RemoteModel:
        def decide(self, text, criterion):
            """Decide.
            
            Args:
                text: Input post as a string, without reference labels. Treated as untrusted evidence, not classifier instructions.
                criterion: Input criterion.
            """
            try:
                result = client.ask(state=text, questions={"moderation": criterion}, model=model_alias)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Inference server HTTP {exc.code}: {detail}") from exc
            if expected_model is not None and result.get("model") != expected_model:
                raise ValueError("Server model identity differs from the initial health check")
            answer = dict(result["answers"]["moderation"])
            answer["input_tokens"] = result.get("usage", {}).get("input_tokens")
            answer["returned_model"] = result.get("model")
            return answer

    return make_choice_predictor(RemoteModel(), question, positive_class=positive_class,
                                 negative_class=negative_class, confidence_is_probability=False)


def multiclass_metrics(predictions, reference_labels, classes):
    """Native multiclass metrics on successful rows, aligned with all input rows.
    
    Args:
        predictions: Prediction records aligned one-to-one with the reference rows; for internal binary metrics, the successful-row binary decisions.
        reference_labels: Ground-truth class names aligned with every prediction, including failed rows.
        classes: Ordered class names, e.g. ["normal", "offensive", "hatespeech"]; order determines confusion-matrix rows and columns.
    """
    classes = list(classes)
    if not classes or len(classes) != len(set(classes)) or len(predictions) != len(reference_labels):
        raise ValueError("Classes must be unique and references aligned with predictions")
    indices = {name: i for i, name in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    scored = 0
    for row, actual in zip(predictions, reference_labels):
        if actual not in indices:
            raise ValueError(f"Unknown reference class {actual}")
        if row.get("error") is not None:
            continue
        predicted = row["predicted_class"]
        if predicted not in indices:
            raise ValueError(f"Unknown predicted class {predicted}")
        matrix[indices[actual]][indices[predicted]] += 1
        scored += 1
    if not scored:
        raise ValueError("No successful multiclass predictions")
    per_class = {}
    for name, i in indices.items():
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        per_class[name] = {"precision": tp / (tp + fp) if tp + fp else 0.,
                           "recall": tp / (tp + fn) if tp + fn else 0.,
                           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                           "support": sum(matrix[i])}
    return {"classes": classes, "confusion_matrix": matrix, "scored": scored,
            "backend_errors": len(predictions) - scored, "coverage": scored / len(predictions),
            "accuracy": sum(matrix[i][i] for i in range(len(classes))) / scored,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(classes),
            "per_class": per_class, "undefined_metric_policy": "zero for per-class zero denominators"}


def predict_split(data, predict, output_dir, *, split, text_column="text", label_column="label",
                  metadata=None, synchronize=None, progress_every=25):
    """Run any text-only predictor and persist one split; never pass labels to it.
    
    Optional synchronize() brackets measured GPU calls. Backend errors are kept as
    missing decisions. Invalid source labels fail before any inference starts.
    Existing output directories are not overwritten.
    
    Args:
        data: Iterable of records with text and integer binary labels; labels are saved for evaluation and never passed to the predictor.
        predict: Callable accepting only a text string and returning a normalized prediction dictionary with p_positive, predicted_label, and confidence.
        output_dir: Destination folder for downloaded data or saved reports, depending on the operation.
        split: Split name recorded in the manifest, normally train, validation, or test.
        text_column: Dataset text column name (default "text"); trimmed text is used for overlap checks.
        label_column: Ground-truth binary label column (default "label"), containing integer 0/1 values.
        metadata: JSON-serializable model, rubric, and dataset configuration; must match between train and held-out threshold comparisons.
        synchronize: Optional zero-argument GPU synchronization callback run around measured prediction calls; None uses ordinary wall-clock timing.
        progress_every: Print progress every this many rows; 0 disables progress messages (default 25).
    """
    examples = []
    for index, item in enumerate(data):
        text, label = item[text_column], item[label_column]
        if not isinstance(text, str) or type(label) is not int or label not in (0, 1):
            raise ValueError(f"Invalid text or binary label at row {index}")
        examples.append((text, label))
    if not examples:
        raise ValueError("No examples supplied")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"split": split, "examples": len(examples), "metadata": metadata or {},
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "text_column": text_column, "label_column": label_column,
                "fingerprint": getattr(data, "_fingerprint", None)}
    _write_json(folder / "manifest.json", manifest)
    rows = []
    with (folder / "predictions.jsonl").open("w") as stream:
        for index, (text, label) in enumerate(examples):
            row = {"row_index": index, "split": split, "text": text, "label": label,
                   "text_sha256": _text_hash(text)}
            start = None
            try:
                if synchronize:
                    synchronize()
                start = time.perf_counter()
                result = dict(predict(text))
                if synchronize:
                    synchronize()
                row["latency_ms"] = (time.perf_counter() - start) * 1000
                p = probability(result["p_positive"])
                confidence = probability(result["confidence"])
                if type(result["predicted_label"]) is not int or result["predicted_label"] not in (0, 1):
                    raise ValueError("Predictor must return integer predicted_label 0/1")
                # Serialize before publishing the result, rejecting invalid native values.
                json.dumps(result, allow_nan=False)
                row.update({"p_positive": p, "confidence": confidence,
                            "predicted_label": result["predicted_label"],
                            "predicted_class": result.get("predicted_class"),
                            "selected_class_probability": result.get("selected_class_probability"),
                            "input_tokens": result.get("input_tokens"), "raw_answer": result.get("raw_answer"),
                            "correct": result["predicted_label"] == label, "error": None})
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", p_positive=None,
                           predicted_label=None, confidence=None, correct=None)
                if start is not None:
                    row["latency_ms"] = (time.perf_counter() - start) * 1000
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(examples)):
                print(f"{split}: {index + 1}/{len(examples)}; errors={sum(r['error'] is not None for r in rows)}")
    manifest.update(status="completed_with_errors" if any(r["error"] for r in rows) else "completed",
                    finished_at=datetime.now(timezone.utc).isoformat())
    _write_json(folder / "manifest.json", manifest)
    return rows


def fit_training_threshold(train_dir):
    """Fit only a saved train split and freeze a reusable threshold artifact.
    
    Args:
        train_dir: Completed train prediction folder with both binary labels present; writes frozen_threshold.json and accuracy analysis.
    """
    folder = Path(train_dir)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["split"] != "train" or manifest["status"] not in ("completed", "completed_with_errors"):
        raise ValueError("Threshold fitting requires completed train predictions")
    rows = _read_rows(folder)
    if {r["label"] for r in rows if r.get("error") is None} != {0, 1}:
        raise ValueError("Training threshold fitting needs successful examples from both classes")
    report = optimize_accuracy_threshold(folder, score_column="p_positive")
    artifact = {"threshold": report["best"]["threshold"], "selection_split": "train",
                "objective": "maximum accuracy (minimum FP + FN)", "decision_rule": "p_positive >= threshold",
                "train_dir": str(folder.resolve()), "metadata": manifest["metadata"],
                "training_predictions_sha256": hashlib.sha256((folder / "predictions.jsonl").read_bytes()).hexdigest(),
                "training_text_hashes": sorted({r["text_sha256"] for r in rows}),
                "training_scored": report["scored"], "training_excluded_errors": report["excluded_errors"],
                "training_best": report["best"], "training_baseline": report["baseline"],
                "created_at": datetime.now(timezone.utc).isoformat()}
    _write_json(folder / "frozen_threshold.json", artifact)
    return artifact


def _classification_metrics(rows, predictions):
    """Classification metrics.
    
    Args:
        rows: Prediction records for one task/split; records marked with errors are excluded from scored metrics.
        predictions: Prediction records aligned one-to-one with the reference rows; for internal binary metrics, the successful-row binary decisions.
    """
    valid = [r for r in rows if r.get("error") is None]
    if not valid:
        raise ValueError("No successful test predictions")
    tp = sum(r["label"] == 1 and p == 1 for r, p in zip(valid, predictions))
    fp = sum(r["label"] == 0 and p == 1 for r, p in zip(valid, predictions))
    tn = sum(r["label"] == 0 and p == 0 for r, p in zip(valid, predictions))
    fn = sum(r["label"] == 1 and p == 0 for r, p in zip(valid, predictions))
    def ratio(a, b):
        """Ratio.
        
        Args:
            a: Numerator of the ratio.
            b: Denominator; zero produces an undefined metric instead of division.
        """
        return a / b if b else None
    latencies = sorted(r["latency_ms"] for r in valid)
    return {"attempted": len(rows), "scored": len(valid), "backend_errors": len(rows) - len(valid),
            "coverage": len(valid) / len(rows), "accuracy": (tp + tn) / len(valid),
            "errors": fp + fn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "benign_block_rate": ratio(fp, fp + tn),
            "brier_score": sum((r["p_positive"] - r["label"]) ** 2 for r in valid) / len(valid),
            "mean_native_confidence": sum(r["confidence"] for r in valid) / len(valid),
            "latency_p50_ms": latencies[math.ceil(.5 * len(valid)) - 1],
            "latency_p95_ms": latencies[math.ceil(.95 * len(valid)) - 1]}


def compare_test_thresholds(test_dir, threshold_file, *, evaluation_split="test"):
    """Apply a frozen train threshold to held-out scores; never refit on their labels.
    
    Set evaluation_split="validation" to evaluate validation predictions.
    
    Compares native model choices, threshold 0.5, and the train-selected threshold
    on the same successful test rows. Returns the summary and per-example changes.
    
    Args:
        test_dir: Folder containing saved held-out predictions and manifest; can contain validation when evaluation_split="validation".
        threshold_file: Path to a frozen_threshold.json produced from training predictions; checked against held-out metadata and text overlap.
        evaluation_split: "test" (default) or "validation". Must match the saved prediction manifest; neither option refits a threshold.
    """
    folder = Path(test_dir)
    manifest = json.loads((folder / "manifest.json").read_text())
    artifact = json.loads(Path(threshold_file).read_text())
    if evaluation_split not in ("validation", "test"):
        raise ValueError("evaluation_split must be validation or test")
    if manifest["split"] != evaluation_split or artifact["selection_split"] != "train":
        raise ValueError("Expected test predictions and a train-selected threshold")
    if manifest["status"] not in ("completed", "completed_with_errors"):
        raise ValueError("Test predictions are incomplete")
    if manifest["metadata"] != artifact["metadata"]:
        raise ValueError("Train/test model, question, or dataset metadata differs")
    threshold = artifact["threshold"]
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError("Invalid frozen threshold")
    rows = _read_rows(folder)
    if set(artifact["training_text_hashes"]) & {r["text_sha256"] for r in rows}:
        raise ValueError("Training/test texts overlap; refusing a leaked comparison")
    valid = [r for r in rows if r.get("error") is None]
    native = [r["predicted_label"] for r in valid]
    baseline = [int(probability(r["p_positive"]) >= .5) for r in valid]
    tuned = [int(probability(r["p_positive"]) >= threshold) for r in valid]
    comparison = {"native": _classification_metrics(rows, native),
                  "threshold_0.5": _classification_metrics(rows, baseline),
                  "train_threshold": _classification_metrics(rows, tuned)}
    changed = []
    for row, before, after in zip(valid, native, tuned):
        changed.append({**row, "native_label": before, "tuned_label": after,
                        "native_correct": before == row["label"], "tuned_correct": after == row["label"],
                        "changed": before != after})
    report = {"selection_split": "train", "evaluation_split": evaluation_split, "threshold": threshold,
              "threshold_artifact": str(Path(threshold_file).resolve()), "comparison": comparison,
              "accuracy_gain_percentage_points": 100 * (comparison["train_threshold"]["accuracy"] - comparison["native"]["accuracy"]),
              "errors_removed": comparison["native"]["errors"] - comparison["train_threshold"]["errors"],
              "fixed": sum(not r["native_correct"] and r["tuned_correct"] for r in changed),
              "broken": sum(r["native_correct"] and not r["tuned_correct"] for r in changed),
              "note": "Frozen training threshold; test labels used for metrics only. Brier score is unchanged because probabilities are unchanged."}
    _write_json(folder / "threshold_comparison.json", report)
    with (folder / "threshold_comparison.jsonl").open("w") as stream:
        for row in changed:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return report, changed
