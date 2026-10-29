"""Download evaluation datasets as offline Hugging Face DatasetDict snapshots."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


CATALOG = {
    "deepset": {"repo": "deepset/prompt-injections", "config": None,
                "text": "text", "labels": ["label"]},
    "wildguard": {"repo": "allenai/wildguardmix", "config": "wildguardtest",
                  "text": "prompt", "labels": ["prompt_harm_label"]},
    "toxicchat": {"repo": "lmsys/toxic-chat", "config": "toxicchat0124",
                  "text": "user_input", "labels": ["toxicity", "jailbreaking"]},
    "xstest": {"repo": "Paul/XSTest", "config": None,
               "text": "prompt", "labels": ["label", "type"]},
    "hatexplain": {"repo": "Hate-speech-CNERG/hatexplain", "config": "plain_text",
                   "text": "post_tokens", "labels": ["annotators"]},
    "moderation": {"repo": "mmathys/openai-moderation-api-evaluation", "config": None,
                   "text": "prompt", "labels": ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]},
}


def load_hatexplain(revision, cache_dir):
    """Use the original author's JSON, avoiding HF's legacy executable loader.
    
    Hugging Face datasets still handles downloads, Arrow construction, and saving.
    All per-annotator labels and official split memberships remain untouched.
    
    Args:
        revision: Requested dataset Git/Hugging Face revision; None uses the downloader default.
        cache_dir: Local download cache directory path.
    """
    from datasets import Dataset, DatasetDict, DownloadConfig, DownloadManager

    request = Request(
        f"https://api.github.com/repos/hate-alert/HateXplain/commits/{revision}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "jev-guardrails"},
    )
    with urlopen(request, timeout=60) as response:
        commit = json.load(response)["sha"]
    base = f"https://raw.githubusercontent.com/hate-alert/HateXplain/{commit}/Data/"
    urls = {"records": base + "dataset.json", "splits": base + "post_id_divisions.json"}
    manager = DownloadManager(download_config=DownloadConfig(cache_dir=str(cache_dir)))
    paths = manager.download(urls)
    records = json.loads(Path(paths["records"]).read_text())
    divisions = json.loads(Path(paths["splits"]).read_text())
    seen = set()
    result = DatasetDict()
    for original, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        ids = divisions[original]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("HateXplain contains duplicate IDs across/within splits")
        seen.update(ids)
        result[split] = Dataset.from_list([
            {"id": key, "annotators": records[key]["annotators"],
             "rationales": records[key]["rationales"], "post_tokens": records[key]["post_tokens"]}
            for key in ids
        ])
    provenance = {"source": "https://github.com/hate-alert/HateXplain", "revision": commit,
                  "loader": "datasets.DownloadManager + Dataset.from_list",
                  "files": {key: {"url": urls[key], "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                            for key, path in paths.items()},
                  "label_names": {"0": "hatespeech", "1": "normal", "2": "offensive"}}
    return result, provenance


def download_one(name, output_dir, revision=None):
    """Download one.
    
    Args:
        name: Dataset key from the downloader registry, or configuration variable name when resolving a required setting.
        output_dir: Destination folder for downloaded data or saved reports, depending on the operation.
        revision: Requested dataset Git/Hugging Face revision; None uses the downloader default.
    """
    from datasets import DatasetDict, load_dataset, load_from_disk
    from huggingface_hub import HfApi

    spec = CATALOG[name]
    destination = output_dir / name
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new output directory")
    cache_dir = output_dir / ".hf-cache"
    if name == "hatexplain":
        data, source = load_hatexplain(revision or "master", cache_dir)
    else:
        # Resolve moving branch names once, then load by immutable commit SHA.
        info = HfApi().dataset_info(spec["repo"], revision=revision or "main")
        if not info.sha:
            raise ValueError("Hugging Face did not return a dataset revision")
        data = load_dataset(spec["repo"], name=spec["config"], revision=info.sha,
                            cache_dir=str(cache_dir))
        source = {"source": f"https://huggingface.co/datasets/{spec['repo']}",
                  "revision": info.sha, "config": spec["config"], "loader": "datasets.load_dataset"}
    if not isinstance(data, DatasetDict) or not data:
        raise ValueError("Expected a nonempty DatasetDict")
    required = {spec["text"], *spec["labels"]}
    for split, table in data.items():
        if not len(table) or not required.issubset(table.column_names):
            raise ValueError(f"{name}/{split}: empty split or missing columns {required - set(table.column_names)}")
    manifest = {
        "dataset": name, "catalog_repository": spec["repo"], **source,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "datasets_version": importlib.metadata.version("datasets"),
        "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
        "text_column": spec["text"], "label_columns": spec["labels"],
        "splits": {split: {"rows": len(table), "features": table.features.to_dict(),
                            "fingerprint": table._fingerprint} for split, table in data.items()},
        "note": "Original labels retained. Missing labels remain missing. No evaluation split was invented.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish only a fully saved and reloadable snapshot. Failures leave no final directory.
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output_dir) as temporary:
        snapshot = Path(temporary) / "snapshot"
        data.save_to_disk(str(snapshot))
        restored = load_from_disk(str(snapshot))
        if {k: len(v) for k, v in restored.items()} != {k: len(v) for k, v in data.items()}:
            raise ValueError("Saved dataset failed row-count verification")
        (snapshot / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        snapshot.rename(destination)
    print(f"Saved {name}: " + ", ".join(f"{s}={len(t)}" for s, t in data.items()), flush=True)
    return destination


def main(argv=None):
    """Main.
    
    Args:
        argv: Optional command-line argument list excluding the program name; None reads sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["all", *CATALOG], default=["deepset"],
                        help="Default: deepset. Use all for the six shortlisted datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/downloads/datasets"))
    parser.add_argument("--revision", help="Pin a revision when downloading one dataset; Git commit for HateXplain")
    parser.add_argument("--list", action="store_true", help="List sources without downloading")
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in CATALOG.items():
            print(f"{name:12} {spec['repo']}  config={spec['config'] or 'default'}")
        return 0
    names = list(CATALOG) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    if args.revision and len(names) != 1:
        parser.error("--revision requires exactly one dataset")
    try:
        import datasets  # noqa: F401
    except ImportError:
        parser.exit(2, 'The selected Python environment needs datasets; see docs/environment.md.\n')
    failures = []
    for name in names:
        try:
            download_one(name, args.output_dir, args.revision)
        except Exception as exc:
            failures.append(name)
            print(f"FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            if name == "wildguard":
                print("Accept the dataset conditions on HF, then authenticate with hf auth login.")
    if failures:
        print("Not downloaded: " + ", ".join(failures))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
