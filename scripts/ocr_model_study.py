"""Compare compatible OCR weights in RTen while holding extraction rules fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tomllib
import zipfile

import financial_benchmark as f
from download_clip import digest, download

CRATE, ROOT = f.CRATE, f.ROOT
DATA = CRATE / "models/ocr-model-study"
SAMPLES = CRATE / "model-manifests/ocr-model-samples.json"
CANDIDATES = CRATE / "model-manifests/ocr-model-candidates.json"
SELECTION = CRATE / "model-manifests/ocr-model-selection.json"
REPORT = CRATE / "docs/ocr-model-results.json"
DOMAINS = ("receipt", "invoice", "cheque")


def models():
    return {c["id"]: c for c in f.read(CANDIDATES)["candidates"]}


def model_dir(candidate):
    return ROOT / candidate["directory"] if "directory" in candidate else DATA / "packs" / candidate["id"]


def download_models():
    candidates = models()
    # Small precedes Latin because the latter reuses its detector.
    for name in ("tiny", "small", "medium", "latin"):
        candidate = candidates[name]
        archive = candidate["archive"]
        download(DATA / "packs", archive, archive["url"])
        directory = model_dir(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DATA / "packs" / archive["name"]) as z:
            for artifact in candidate["files"]:
                if "copied_from" in artifact:
                    data = (DATA / "packs" / artifact["copied_from"]).read_bytes()
                else:
                    member = artifact.get("member") or next(p for p in z.namelist() if Path(p).name == artifact["name"])
                    data = z.read(member)
                if len(data) != artifact["size"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                    raise ValueError("archive member does not match pinned model")
                (directory / artifact["name"]).write_bytes(data)
    print("OCR_MODEL_DOWNLOAD_VERIFIED")


def download_data():
    archives = {a["path"]: a for a in f.read(f.MANIFEST)["sources"]["zenodo"]["archives"]}
    verified_archives = set()
    for sample in f.read(SAMPLES)["samples"]:
        directory = ROOT / Path(sample["image_path"]).parent
        directory.mkdir(parents=True, exist_ok=True)
        for artifact in sample["artifacts"]:
            if "url" in artifact:
                download(directory, dict(artifact, name=artifact["path"]), artifact["url"])
            else:
                archive = archives[artifact["archive"]]
                if archive["path"] not in verified_archives:
                    download(f.DATA, dict(archive, name=archive["path"]), archive["url"])
                    verified_archives.add(archive["path"])
                with zipfile.ZipFile(f.DATA / archive["path"]) as z:
                    data = z.read(artifact["member"])
                if len(data) != artifact["size"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                    raise ValueError("archive member does not match sample checksum")
                (directory / artifact["path"]).write_bytes(data)
    print("OCR_MODEL_SAMPLES_DOWNLOADED")


def available(field, label, sample, row):
    if field != "date":
        return f.target_in_ocr(field, label, sample, row)
    expected = f.date_key(label, sample["date_order"])
    text = " ".join(s["text"] for s in row.get("spans", []))
    # Diagnostic only: find equivalent printed forms without repairing letters.
    patterns = (r"[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}",
                r"[0-9]{1,2}/[0-9]{1,2}/(?:[0-9]{4}|[0-9]{2})",
                r"[0-9]{1,2}-[0-9]{1,2}-(?:[0-9]{4}|[0-9]{2})",
                r"[0-9]{1,2}\s+[A-Za-z]{3,9}\s+(?:[0-9]{4}|[0-9]{2})",
                r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}")
    return any(f.date_key(m, sample["date_order"]) == expected for pattern in patterns for m in re.findall(pattern, text))


def metrics(samples, rows):
    field_metrics = f.score(samples, rows)
    present = total = 0
    by_field = {}
    for sample in samples:
        row = rows[sample["id"]]
        for field, label in sample["expected"].items():
            if label is None:
                continue
            found = available(field, label, sample, row)
            present += found
            total += 1
            record = by_field.setdefault(field, {"present": 0, "total": 0})
            record["present"] += found
            record["total"] += 1
    times = [rows[s["id"]]["elapsed_ms"] for s in samples if "elapsed_ms" in rows[s["id"]]]
    return {"extraction": field_metrics,
            "target_availability": {"present": present, "total": total, "recall": present / total if total else 0, "fields": by_field},
            "observed_document_ms": {"median": statistics.median(times) if times else None, "total": sum(times)}}


def choose(candidates, results):
    # Defined before observing candidate metrics. Memory breaks quality ties.
    return max(results, key=lambda name: (results[name]["target_availability"]["present"],
                                        results[name]["extraction"]["aggregate"]["correct"],
                                        -sum(a["size"] for a in candidates[name]["files"]), name))


def verify_data():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]
    if workspace["rten"]["version"] != "=0.27.0" or workspace["rusto"]["version"] != "=0.3.2":
        raise ValueError("RTen/RustO versions changed during model selection")
    manifest = f.read(SAMPLES)
    f.validate_manifest(manifest)
    previous = f.read(f.MANIFEST)
    seen_ids = {s["id"] for s in previous["samples"]}
    seen_images = {a["sha256"] for s in previous["samples"] for a in s["artifacts"] if a["path"] == s["image"]}
    for s in manifest["samples"]:
        directory = ROOT / Path(s["image_path"]).parent
        for a in s["artifacts"]:
            p = directory / a["path"]
            if p.stat().st_size != a["size"] or digest(p) != a["sha256"]:
                raise ValueError(f"sample bytes changed: {s['id']}")
        if s["split"] == "test":
            if s["id"] in seen_ids or digest(ROOT / s["image_path"]) in seen_images:
                raise ValueError("fresh evaluation overlaps the prior study")
    for c in models().values():
        directory = model_dir(c)
        for a in c["files"]:
            p = directory / a["name"]
            if p.stat().st_size != a["size"] or digest(p) != a["sha256"]:
                raise ValueError(f"model bytes changed: {c['id']}")
    previous_sources = f.read(f.PROFILE)["sources"]
    f.verify_sources(previous_sources)
    print("OCR_MODEL_DATA_VERIFIED")


def run(candidate_name, split, domains):
    if f.code_hashes() != f.read(f.PROFILE)["sources"]:
        raise ValueError("frozen extraction implementation changed; start a new study")
    candidate = models()[candidate_name]
    samples = [s for s in f.read(SAMPLES)["samples"] if s["split"] == split and s["domain"] in domains]
    if split == "test":
        frozen = f.read(SELECTION)
        if digest(SAMPLES) != frozen["samples_sha256"] or digest(CANDIDATES) != frozen["candidates_sha256"]:
            raise ValueError("study configuration changed after selection")
        if digest(CRATE / "examples/ocr_model_batch.rs") != frozen["batch_source_sha256"]:
            raise ValueError("Rust inference code changed after selection")
    jobs = [{"id": s["id"], "kind": s["kind"], "image": s["image_path"], "decimal_separator": s["decimal_separator"]} for s in samples]
    command = ["rtk", "proxy", "mise", "exec", "--", "cargo", "run", "-p", "rs-ocr", "--release", "--locked", "--example", "ocr_model_batch", "--", str(model_dir(candidate)), candidate["preset"]]
    output = DATA / f"{split}-{candidate_name}.jsonl"
    if output.exists() and split == "test":
        raise ValueError("refusing to overwrite existing held-out predictions")
    with output.open("w") as destination:
        subprocess.run(command, cwd=ROOT, input="".join(json.dumps(j) + "\n" for j in jobs), text=True, stdout=destination, check=True)
    rows = f.predictions(output)
    if set(rows) != {s["id"] for s in samples}:
        raise ValueError("incomplete inference batch")
    if any("error" in row for row in rows.values()):
        raise ValueError("candidate inference errors: inspect saved evidence")
    print(f"OCR_MODEL_BATCH_COMPLETED {split} {candidate_name}: {len(rows)}")


def development():
    manifest = f.read(SAMPLES)
    candidates = models()
    result = {domain: {} for domain in DOMAINS}
    for name in candidates:
        rows = f.predictions(DATA / f"dev-{name}.jsonl")
        expected = {s["id"] for s in manifest["samples"] if s["split"] == "dev"}
        if set(rows) != expected or any("error" in row for row in rows.values()):
            raise ValueError("incomplete or failed development candidate")
        for domain in DOMAINS:
            samples = [s for s in manifest["samples"] if s["split"] == "dev" and s["domain"] == domain]
            result[domain][name] = metrics(samples, rows)
    return result


def freeze():
    if any(DATA.glob("test-*.jsonl")):
        raise ValueError("held-out results already exist; use a new study for another selection")
    results = development()
    f.write(SELECTION, {"criterion": "maximize development positive target availability; tie-break by correct extracted fields, then smaller model pack", "selected": {domain: choose(models(), results[domain]) for domain in DOMAINS},
                        "samples_sha256": digest(SAMPLES), "candidates_sha256": digest(CANDIDATES), "batch_source_sha256": digest(CRATE / "examples/ocr_model_batch.rs"),
                        "development_prediction_sha256": {name: digest(DATA / f"dev-{name}.jsonl") for name in models()}})
    print("OCR_MODEL_SELECTION_FROZEN", f.read(SELECTION)["selected"])


def report():
    selection = f.read(SELECTION)
    manifest = f.read(SAMPLES)
    if digest(SAMPLES) != selection["samples_sha256"] or digest(CANDIDATES) != selection["candidates_sha256"]:
        raise ValueError("selection configuration changed")
    if digest(CRATE / "examples/ocr_model_batch.rs") != selection["batch_source_sha256"]:
        raise ValueError("Rust model evaluation changed after selection")
    dev = development()
    for domain in DOMAINS:
        if selection["selected"][domain] != choose(models(), dev[domain]):
            raise ValueError("selection does not match development criterion")
    for name, sha in selection["development_prediction_sha256"].items():
        if digest(DATA / f"dev-{name}.jsonl") != sha:
            raise ValueError("development evidence changed after freeze")
    test = {}
    for domain in DOMAINS:
        selected = selection["selected"][domain]
        samples = [s for s in manifest["samples"] if s["split"] == "test" and s["domain"] == domain]
        test[domain] = {}
        for name in sorted({"tiny", selected}):
            rows = f.predictions(DATA / f"test-{name}.jsonl")
            if not {s["id"] for s in samples} <= set(rows) or any("error" in rows[s["id"]] for s in samples):
                raise ValueError("held-out inference failed or incomplete")
            test[domain][name] = metrics(samples, rows)
    prior = f.read(SAMPLES)
    if digest(f.REPORT) != prior["prior_report_sha256"] or digest(f.MANIFEST) != prior["prior_manifest_sha256"]:
        raise ValueError("prior study artifacts changed")
    files = sorted(DATA.glob("dev-*.jsonl")) + sorted(DATA.glob("test-*.jsonl"))
    return {"version": 1, "runtime": "RTen 0.27.0 via RustO 0.3.2, CPU FP32", "protocol": manifest["protocol"], "selection": selection,
            "prediction_sha256": {p.name: digest(p) for p in files}, "development": dev, "test": test,
            "limits": ["Best among four tested packs for this domain and small sample; no universal best-model claim.",
                       "Target availability ignores layout and can find a matching value in another field; it does not prove field understanding.",
                       "All model selection uses development samples only. Fresh test images share no IDs or source image hashes with the prior financial study.",
                       "Cheque annotations are manual and not blinded; historical scans are not representative of modern banking.",
                       "Observed CPU times include OCR and extraction; the first document includes model loading. Timings are not a controlled performance benchmark.",
                       "Original OCR weights, field parser and prior financial report remain untouched; Python scores results but does not perform inference."]}


def regressions():
    commands = [["python", str(CRATE / "scripts/financial_benchmark.py"), "verify-report"],
                ["cargo", "clippy", "-p", "rs-ocr", "--all-features", "--all-targets", "--locked", "--", "-D", "warnings"],
                ["cargo", "fmt", "-p", "rs-ocr", "--", "--check"]]
    for command in commands:
        subprocess.run(["rtk", "proxy", "mise", "exec", "--", *command], cwd=ROOT, check=True)
    print("OCR_MODEL_REGRESSIONS_VERIFIED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("download-models", "download-data", "run", "freeze", "report", "verify-data", "verify-report", "verify-regressions"))
    p.add_argument("--model", choices=tuple(models()))
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    args = p.parse_args()
    if args.command == "download-models":
        download_models()
    elif args.command == "download-data":
        download_data()
    elif args.command == "run":
        if not args.model:
            p.error("run requires --model")
        run(args.model, args.split, args.domains)
    elif args.command == "freeze":
        freeze()
    elif args.command == "verify-data":
        verify_data()
    elif args.command == "verify-regressions":
        regressions()
    elif args.command == "report":
        f.write(REPORT, report())
        print("OCR_MODEL_REPORT_WRITTEN")
    else:
        if f.read(REPORT) != report():
            raise ValueError("stale or altered model study report")
        print("OCR_MODEL_REPORT_VERIFIED")


if __name__ == "__main__":
    main()
