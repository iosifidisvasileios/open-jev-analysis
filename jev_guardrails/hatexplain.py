"""Prepare original HateXplain annotator records for binary and multiclass evals."""

from collections import Counter


def prepare_hatexplain(dataset):
    """Return a new HF DatasetDict and an audit; source data is never modified.
    
    Resolve the original three-class majority before grouping offensive/hatespeech
    as positive. Unresolved votes are excluded, never silently mapped to normal.
    Remove training texts also found in test (whitespace-trimmed exact matches),
    retaining the labeled test split intact. Test labels are not used for cleanup.
    
    Args:
        dataset: Hugging Face dataset collection containing official splits and required source columns.
    """
    from datasets import Dataset, DatasetDict

    label_names = {0: "hatespeech", 1: "normal", 2: "offensive"}
    prepared, audit = {}, {"splits": {}, "binary_positive": ["offensive", "hatespeech"],
                           "vote_policy": "unique majority of original three-class annotator labels"}
    for split, table in dataset.items():
        rows, ambiguous = [], 0
        for row in table:
            annotations = row["annotators"]
            votes = annotations["label"] if isinstance(annotations, dict) else [a["label"] for a in annotations]
            votes = [label_names.get(v, v) for v in votes]
            if not votes or not set(votes) <= set(label_names.values()):
                raise ValueError(f"Unexpected annotator labels in {row['id']}")
            counts = Counter(votes)
            winners = [key for key, count in counts.items() if count == max(counts.values())]
            if len(winners) != 1 or max(counts.values()) <= len(votes) / 2:
                ambiguous += 1
                continue
            text = " ".join(row["post_tokens"])
            if not text.strip():
                raise ValueError(f"Empty post {row['id']}")
            rows.append({"id": row["id"], "text": text, "label_multiclass": winners[0],
                         "label": int(winners[0] != "normal"), "annotator_labels": votes})
        prepared[split] = rows
        audit["splits"][split] = {"source_rows": len(table), "ambiguous_excluded": ambiguous}
    test_texts = {r["text"].strip() for r in prepared["test"]}
    removed = [r for r in prepared["train"] if r["text"].strip() in test_texts]
    prepared["train"] = [r for r in prepared["train"] if r["text"].strip() not in test_texts]
    audit["train_test_overlap"] = {"removed_training_rows": len(removed), "removed_training_ids": [r["id"] for r in removed]}
    for split, rows in prepared.items():
        if not rows:
            raise ValueError(f"No usable {split} examples")
        audit["splits"][split].update(prepared_rows=len(rows),
                                     multiclass_counts=dict(Counter(r["label_multiclass"] for r in rows)))
    return DatasetDict({split: Dataset.from_list(rows) for split, rows in prepared.items()}), audit
