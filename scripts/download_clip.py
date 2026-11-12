"""Fetch pinned CLIP files; inference itself has no Python dependency."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def download(directory, artifact, url):
    name = artifact["name"]
    if Path(name).name != name:
        raise ValueError("artifact name must be a filename")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if destination.is_file() and destination.stat().st_size == artifact["size"]:
        if digest(destination) == artifact["sha256"]:
            print(f"Verified cached {destination}")
            return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
            result = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > artifact["size"]:
                        raise ValueError(f"unexpected size for {name}")
                    output.write(chunk)
                    result.update(chunk)
        if size != artifact["size"] or result.hexdigest() != artifact["sha256"]:
            raise ValueError(f"checksum/size mismatch for {name}")
        temporary.replace(destination)
        print(f"Downloaded and verified {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    crate = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=crate / "models/clip-vit-b32")
    parser.add_argument("--test-images", action="store_true", help="also fetch upstream reference photos")
    args = parser.parse_args()
    manifest = json.loads((crate / "model-manifests/clip-vit-b32.json").read_text())
    for artifact in manifest["files"]:
        download(args.output, artifact, manifest["base_url"] + artifact["remote"])
    if args.test_images:
        for artifact in manifest["test_images"]:
            download(args.output / "test-images", artifact, artifact["url"])
    (args.output / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
