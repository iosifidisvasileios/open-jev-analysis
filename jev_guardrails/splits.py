"""Freeze and verify official train/test subsets shared by all model notebooks."""

import hashlib
import json
from pathlib import Path


def _digest(value):
    """Digest.
    
    Args:
        value: Value to validate, hash, or serialize; probability() requires a finite numeric value in [0, 1].
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_fixed_splits(dataset, *, profile, project_root, source_revision):
    """Select official split prefixes using one shared config and frozen manifest.
    
    No randomness or model input participates. Initial creation freezes selected
    row identities/order and full prepared split contents. Later calls verify all
    of those, the source revision, and configured limits. Changes fail explicitly;
    choose a new manifest_name/profile for an intentional new benchmark subset.
    
    Args:
        dataset: Hugging Face dataset collection containing official splits and required source columns.
        profile: Dataset profile key in config/dataset_splits.json; selects shared limits and frozen manifest.
        project_root: Project root path containing config/ and data/split_manifests/.
        source_revision: Exact dataset source revision recorded in the fixed split manifest; mismatches are rejected.
    """
    root = Path(project_root)
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("A nonempty source_revision is required")
    config = json.loads((root / "config/dataset_splits.json").read_text())[profile]
    name = config["manifest_name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("manifest_name must be a simple filename stem")
    selected, specs = {}, {}
    for split in ("train", "test"):
        table = dataset[split]
        if not len(table):
            raise ValueError(f"Empty {split} split")
        limit = config[f"{split}_limit"]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"Invalid {split}_limit")
        count = len(table) if limit is None else min(limit, len(table))
        selected[split] = table.select(range(count))
        # Canonical content hashes work even for datasets with no explicit IDs.
        full = [_digest(row) for row in table]
        specs[split] = {
            "available_rows": len(table), "limit": limit, "selected_rows": count,
            "prepared_order_sha256": _digest(full),
            "selected_row_sha256": full[:count],
            "selected_ids": [str(row["id"]) if "id" in row else full[i]
                             for i, row in enumerate(selected[split])],
        }
    manifest = {"version": 1, "profile": profile, "source_revision": source_revision,
                "method": "official split membership; original order; first configured N rows; no shuffle",
                "splits": specs}
    manifest["selection_sha256"] = _digest(manifest)
    directory = root / "data/split_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    try:
        with path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        saved = json.loads(path.read_text())
        if saved != manifest:
            raise ValueError(f"Split manifest mismatch: {path}. Dataset contents, order, revision, or limits changed. Restore the original config/data or choose a new manifest_name for an intentional new selection.")
    return selected["train"], selected["test"], manifest
