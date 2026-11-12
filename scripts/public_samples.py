#!/usr/bin/env python3
"""Download/verify a pinned, deterministic offline research sample (stdlib only)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

CRATE = Path(__file__).resolve().parents[1]
MANIFEST = CRATE / "model-manifests/public-selection-samples.json"
DEFAULT_DATA = CRATE / "models/public-selection-samples"
REVISION = "efb3595c33780abf9d326c729973e72d165366c2"
SEED = "rs-ocr-public-evaluation-v1"
BASE = f"https://raw.githubusercontent.com/ssivakorn/reCAPTCHA-study/{REVISION}/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_bytes(data, entry):
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    if len(data) != entry["size"] or hashlib.sha1(blob).hexdigest() != entry["git_blob"]:
        raise ValueError(f"source integrity failure: {entry['path']}")
    if "sha256" in entry and digest(data) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {entry['path']}")


def make_manifest(tree_path):
    tree = json.loads(tree_path.read_text())
    if tree["truncated"]:
        raise ValueError("incomplete upstream file index")
    groups = {}
    for entry in tree["tree"]:
        parts = entry["path"].split("/")
        if len(parts) == 3 and parts[0] == "dataset" and entry["type"] == "blob":
            groups.setdefault(parts[1], {})[parts[2]] = entry
    # Keep shared tile/full-image blobs in one component before splitting.
    parent = {key: key for key in groups}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owner = {}
    for key, files in sorted(groups.items()):
        for name, entry in files.items():
            if name == "full_images.png" or name.startswith("tile_"):
                if entry["sha"] in owner:
                    parent[root(key)] = root(owner[entry["sha"]])
                owner[entry["sha"]] = key
    # One representative per duplicate component, independent of labels/results.
    representatives = {}
    for key in sorted(groups):
        representatives.setdefault(root(key), key)
    buckets = {9: [], 16: []}
    for key in representatives.values():
        count = sum(name.startswith("tile_") for name in groups[key])
        if count not in buckets:
            raise ValueError(f"unsupported tile count in {key}")
        buckets[count].append(key)
    samples = []
    for count, total in [(9, 80), (16, 40)]:
        ordered = sorted(buckets[count], key=lambda key: digest(f"{SEED}:{key}".encode()))
        for index, key in enumerate(ordered[:total]):
            files = groups[key]
            entries = []
            for name in ["full_images.png", "info.json"]:
                item = files[name]
                entries.append({"path": item["path"], "size": item["size"], "git_blob": item["sha"]})
            samples.append({
                "id": key,
                "split": "development" if index < total // 2 else "test",
                "rows": 3 if count == 9 else 4,
                "columns": 3 if count == 9 else 4,
                "tile_blobs": [files[f"tile_{i}.png"]["sha"] for i in range(count)],
                "files": entries,
            })
    manifest = {
        "schema_version": 1,
        "source": "https://github.com/ssivakorn/reCAPTCHA-study",
        "revision": REVISION,
        "license": "CC-BY-NC-4.0",
        "license_url": f"https://github.com/ssivakorn/reCAPTCHA-study/blob/{REVISION}/LICENSE",
        "attribution": "Suphannee Sivakorn, reCAPTCHA Study (2026)",
        "seed": SEED,
        "selection": (
            "SHA-256 order of seed:id; one representative per shared source tile/full-image component; "
            "80 3x3 and 40 4x4; first half of each size for development"
        ),
        "upstream_tree_sha256": digest(tree_path.read_bytes()),
        "samples": sorted(samples, key=lambda sample: (sample["split"], sample["id"])),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"created {len(samples)} sample manifest")


def fetch(entry, directory):
    path = directory / entry["path"]
    if path.exists():
        try:
            verify_bytes(path.read_bytes(), entry)
            return
        except ValueError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(BASE + entry["path"], headers={"User-Agent": "rs-ocr-offline-evaluation/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(entry["size"] + 1)
    verify_bytes(data, entry)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_info(info, count):
    if (not isinstance(info.get("instruction"), str)
            or not isinstance(info.get("keyword"), str) or not info["keyword"].strip()):
        raise ValueError("invalid instruction/keyword")
    answers = info.get("correct_answers")
    if not isinstance(answers, list) or any(type(i) is not int or not 0 <= i < count for i in answers):
        raise ValueError("invalid answers")
    if len(answers) != len(set(answers)):
        raise ValueError("duplicate answer")


def verify(manifest, directory):
    seen_ids = set()
    blobs = {"development": set(), "test": set()}
    counts = {"development": 0, "test": 0}
    for sample in manifest["samples"]:
        if sample["id"] in seen_ids:
            raise ValueError("duplicate sample id")
        seen_ids.add(sample["id"])
        split = sample["split"]
        counts[split] += 1
        blobs[split].update(sample["tile_blobs"])
        for entry in sample["files"]:
            verify_bytes((directory / entry["path"]).read_bytes(), entry)
            if entry["path"].endswith("full_images.png"):
                blobs[split].add(entry["git_blob"])
        info = json.loads((directory / f"dataset/{sample['id']}/info.json").read_text())
        validate_info(info, sample["rows"] * sample["columns"])
    if blobs["development"] & blobs["test"]:
        raise ValueError("identical image bytes cross evaluation split")
    if counts != {"development": 60, "test": 60}:
        raise ValueError(f"unexpected counts: {counts}")
    print("PUBLIC_SAMPLES_VERIFIED", counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create-manifest", "download", "verify"])
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    args.data = args.data.resolve()
    if args.command == "create-manifest":
        if args.tree is None:
            parser.error("create-manifest requires --tree")
        make_manifest(args.tree)
        return
    manifest = json.loads(MANIFEST.read_text())
    if args.command == "download":
        entries = [entry for sample in manifest["samples"] for entry in sample["files"]]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda entry: fetch(entry, args.data), entries))
    verify(manifest, args.data)


if __name__ == "__main__":
    main()
