"""Fetch the pinned detector; model licensing is independent of crate licensing."""

import argparse
import json
from pathlib import Path

from download_clip import download


def main():
    crate = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=crate / "models/yolo11s")
    args = parser.parse_args()
    manifest = json.loads((crate / "model-manifests/yolo11s.json").read_text())
    for artifact in manifest["files"]:
        download(args.output, artifact, manifest["base_url"] + artifact["remote"])
    (args.output / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
