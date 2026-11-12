"""Pinned public financial-document study; all inference/extraction runs in Rust.

Images, transcripts and predictions stay in ignored models/financial-public.
Only provenance, aggregate results and code are suitable for redistribution here.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from download_clip import digest, download

CRATE = Path(__file__).resolve().parent.parent
ROOT = CRATE.parent
DATA = CRATE / "models/financial-public"
MANIFEST = CRATE / "model-manifests/financial-samples.json"
PROFILE = CRATE / "model-manifests/financial-profile.json"
REPORT = CRATE / "docs/financial-results.json"
SOURCES = [CRATE / p for p in ("src/documents/extraction.rs", "src/documents/money.rs", "src/documents/mod.rs", "src/bills.rs", "src/cheques.rs", "examples/financial_batch.rs", "Cargo.toml")]
MONEY = {"amount", "total", "tax", "subtotal"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate prediction ids")
    return result


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}


def verify_sources(sources, *, root=ROOT, archive=DATA / "frozen-sources"):
    """Verify historical evidence against original bytes, never bless new code.

    Execution remains bound to current sources. Only report verification may
    use this archive when subsequent implementations evolve.
    """
    for name, sha in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe historical source path")
        current = root / relative
        original = archive / relative
        if current.is_file() and digest(current) == sha:
            continue
        if not original.is_file() or digest(original) != sha:
            raise ValueError(f"missing or changed frozen source: {name}")


def amount_label(value):
    """Labels sometimes use currency prefixes and different decimal conventions.

    This is annotation decoding, never OCR repair. No float arithmetic/rounding.
    """
    value = re.sub(r"^(?:RM|MYR|USD|EUR|GBP|[$€£])\s*", "", str(value).strip())
    if "," in value and "." not in value:
        value = value.replace(",", ".")
    elif "," in value:
        value = value.replace(",", "")
    try:
        scaled = Decimal(value) * 100
        if not scaled.is_finite() or scaled != scaled.to_integral_value():
            raise ValueError("invalid monetary label")
        return int(scaled)
    except InvalidOperation as exc:
        raise ValueError("invalid monetary label") from exc


def text_key(value):
    # Ignore case/whitespace only. No OCR spelling fixes or fuzzy credit.
    return "".join(str(value).casefold().split())


def date_key(value, order):
    """Normalize printed dates using dataset-declared order, not locale guessing."""
    text = str(value).strip().casefold()
    m = re.fullmatch(r"([0-9]{1,4})[-/]([0-9]{1,2})[-/]([0-9]{2,4})", text)
    if m:
        a, b, c = m.groups()
        if len(a) == 4:
            return f"{int(a):04}-{int(b):02}-{int(c):02}"
        year = int(c) + (2000 if int(c) < 70 else 1900) if len(c) == 2 else int(c)
        day, month = (int(a), int(b)) if order == "dmy" else (int(b), int(a))
        return f"{year:04}-{month:02}-{day:02}"
    months = {name: n for n, name in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
    m = re.fullmatch(r"([0-9]{1,2})\s+([a-z]+)\s+([0-9]{2,4})", text)
    if m and m[2][:3] in months:
        day, month, year = int(m[1]), months[m[2][:3]], int(m[3])
        if len(m[3]) == 2:
            year += 2000 if year < 70 else 1900
        return f"{year:04}-{month:02}-{day:02}"
    m = re.fullmatch(r"([a-z]+)\s+([0-9]{1,2}),?\s+([0-9]{4})", text)
    if m and m[1][:3] in months:
        return f"{int(m[3]):04}-{months[m[1][:3]]:02}-{int(m[2]):02}"
    return text_key(text)


def key(field, value, sample, *, label=False):
    if value is None:
        return None
    if field in MONEY:
        return amount_label(value) if label else value
    if field == "date":
        return date_key(value, sample["date_order"])
    return text_key(value)


def target_in_ocr(field, label, sample, row):
    """Segmentation-insensitive diagnostic, never a substitute for field accuracy.

    Money requires a literal decimal token at the declared precision/convention;
    integer-only handwriting and currency inference are outside this diagnostic.
    """
    text = " ".join(s["text"] for s in row.get("spans", []))
    if field not in MONEY:
        return text_key(label) in text_key(text)
    decimal, grouping = (".", ",") if sample["decimal_separator"] == "dot" else (",", ".")
    pattern = rf"(?<![0-9])[-+]?[0-9]+(?:{re.escape(grouping)}[0-9]{{3}})*{re.escape(decimal)}[0-9]{{2}}(?![0-9])"
    expected = amount_label(label)
    for token in re.findall(pattern, text):
        value = token.replace(grouping, "").replace(decimal, ".")
        if Decimal(value) * 100 == expected:
            return True
    return False


def score(samples, rows):
    counts = defaultdict(Counter)
    exact_documents = 0
    errors = 0
    for sample in samples:
        row = rows[sample["id"]]
        errors += "error" in row
        all_correct = True
        for field, label in sample["expected"].items():
            evidence = row.get("fields", {}).get(field, {})
            value = evidence.get("value")
            expected = key(field, label, sample, label=True)
            actual = key(field, value, sample)
            c = counts[field]
            c["total"] += 1
            correct = actual == expected
            c["correct"] += correct
            c["present"] += value is not None
            c["wrong"] += value is not None and not correct
            c["missing"] += value is None and expected is not None
            c["ambiguous"] += value is None and bool(evidence.get("candidates"))
            c["negative_controls"] += label is None
            c["false_positive"] += label is None and value is not None
            # Diagnostic: whether the target text even exists in OCR. Not an OCR
            # accuracy metric; joining spans ignores segmentation and layout.
            if label is not None and field != "date":
                c["target_in_ocr"] += target_in_ocr(field, label, sample, row)
                c["target_in_ocr_denominator"] += 1
            all_correct &= correct
        exact_documents += all_correct
    totals = Counter()
    for c in counts.values():
        totals.update(c)
    def rates(c):
        return dict(c, accuracy=c["correct"] / c["total"] if c["total"] else 0,
                    coverage=c["present"] / c["total"] if c["total"] else 0)
    return {"documents": len(samples), "exact_documents": exact_documents, "errors": errors,
            "fields": {f: rates(c) for f, c in sorted(counts.items())}, "aggregate": rates(totals)}


def validate_manifest(manifest):
    samples = manifest["samples"]
    ids = [s["id"] for s in samples]
    images = [next(a["sha256"] for a in s["artifacts"] if a["path"] == s["image"]) for s in samples]
    if len(ids) != len(set(ids)) or len(images) != len(set(images)):
        raise ValueError("duplicate sample or image content")
    for domain in ("cheque", "receipt", "invoice"):
        subsets = {split: [s for s in samples if s["domain"] == domain and s["split"] == split] for split in ("dev", "test")}
        if not all(subsets.values()):
            raise ValueError("each domain needs both splits")
        if {s["group"] for s in subsets["dev"]} & {s["group"] for s in subsets["test"]}:
            raise ValueError("issuer group leakage")
    for s in samples:
        if not s["expected"] or s["decimal_separator"] not in ("dot", "comma"):
            raise ValueError("missing labels or invalid locale")
        for field, value in s["expected"].items():
            key(field, value, s, label=True)
        for a in s["artifacts"]:
            if Path(a["path"]).name != a["path"]:
                raise ValueError("unsafe artifact path")


def verify_data(manifest):
    validate_manifest(manifest)
    for s in manifest["samples"]:
        for a in s["artifacts"]:
            path = DATA / a["path"]
            if path.stat().st_size != a["size"] or digest(path) != a["sha256"]:
                raise ValueError(f"checksum mismatch: {path.name}")
        if s["source"] == "sroie":
            labels = read(DATA / f'{s["id"]}.json')
            assert s["expected"] == {k: v for k, v in {"merchant": labels["company"], "date": labels["date"], "total": labels["total"]}.items() if v}
        elif s["source"] == "zenodo":
            labels = read(DATA / f'{s["id"]}.txt')
            expected = {k: labels[n] for k, n in [("merchant", "company"), ("date", "date"), ("total", "total"), ("reference_number", "invoice_number"), ("tax", "iva_amount")] if labels[n]}
            assert s["expected"] == expected
    print("financial data verification passed")


def fetch(manifest):
    validate_manifest(manifest)
    archives = {a["path"]: a for a in manifest["sources"]["zenodo"]["archives"]}
    for s in manifest["samples"]:
        for a in s["artifacts"]:
            if "url" in a:
                download(DATA, dict(a, name=a["path"]), a["url"])
            else:
                archive = archives[a["archive"]]
                download(DATA, dict(archive, name=archive["path"]), archive["url"])
                with zipfile.ZipFile(DATA / archive["path"]) as z:
                    content = z.read(a["member"])
                if len(content) != a["size"] or hashlib.sha256(content).hexdigest() != a["sha256"]:
                    raise ValueError("archive member mismatch")
                (DATA / a["path"]).write_bytes(content)
    verify_data(manifest)


def execute(manifest, split):
    frozen = read(PROFILE)
    if code_hashes() != frozen["sources"]:
        raise ValueError("implementation changed after freeze")
    baseline = predictions(DATA / "baseline.jsonl")
    jobs = [dict(baseline[s["id"]], decimal_separator=s["decimal_separator"]) for s in manifest["samples"] if s["split"] == split]
    command = ["rtk", "proxy", "mise", "exec", "--", "cargo", "run", "-p", "rs-ocr", "--release", "--locked", "--example", "financial_batch", "--", str(CRATE / "models/ppocrv6-tiny")]
    completed = subprocess.run(command, cwd=ROOT, input="".join(json.dumps(j) + "\n" for j in jobs), text=True, stdout=subprocess.PIPE, check=True)
    output = DATA / f"improved-{split}.jsonl"
    output.write_text(completed.stdout)
    if set(predictions(output)) != {j["id"] for j in jobs}:
        raise ValueError("incomplete Rust batch output")
    print(f"Rust extraction evaluated: {split}, {len(jobs)} documents")


def build_report(manifest):
    baseline = predictions(DATA / "baseline.jsonl")
    improved = {}
    for split in ("dev", "test"):
        improved.update(predictions(DATA / f"improved-{split}.jsonl"))
    expected_ids = {s["id"] for s in manifest["samples"]}
    if set(baseline) != expected_ids or set(improved) != expected_ids:
        raise ValueError("prediction/sample mismatch")
    if digest(DATA / "baseline.jsonl") != manifest["baseline"]["evidence_sha256"]:
        raise ValueError("baseline changed")
    for sample in manifest["samples"]:
        a, b = baseline[sample["id"]], improved[sample["id"]]
        if "error" in a or "error" in b:
            raise ValueError("unexpected OCR/extraction errors")
        if a["spans"] != b["spans"]:
            raise ValueError("OCR evidence changed between parser versions")
        for field in b["fields"].values():
            for c in field["candidates"]:
                indices = c["span_indices"]
                if not indices or any(i < 0 or i >= len(b["spans"]) for i in indices):
                    raise ValueError("missing candidate evidence")
                scores = [b["spans"][i]["confidence"] for i in indices]
                expected_score = None if None in scores else min(scores)
                if c["confidence"] != expected_score:
                    raise ValueError("candidate confidence does not match source evidence")
    profile = read(PROFILE)
    verify_sources(profile["sources"])
    if digest(MANIFEST) != profile["manifest_sha256"]:
        raise ValueError("frozen implementation/data changed")
    if any(digest(ROOT / p) != h for p, h in profile["models"].items()):
        raise ValueError("OCR model changed")
    metrics = {}
    for split in ("dev", "test"):
        metrics[split] = {}
        for domain in ("receipt", "invoice", "cheque"):
            samples = [s for s in manifest["samples"] if s["split"] == split and s["domain"] == domain]
            metrics[split][domain] = {"baseline": score(samples, baseline), "improved": score(samples, improved)}
    return {"version": 1, "protocol": manifest["protocol"], "manifest_sha256": digest(MANIFEST), "profile": profile,
            "prediction_sha256": {name: digest(DATA / name) for name in ("baseline.jsonl", "improved-dev.jsonl", "improved-test.jsonl")},
            "metrics": metrics,
            "limitations": ["Small purposive cheque set, including historical handwriting and one blank form; no production banking claim.",
                            "Published labels retained, including a development invoice total that disagrees with its printed discounted payable amount.",
                            "Same author annotated cheque gold fields and improved the parser; not a blinded study. Uncertain cheque fields omitted.",
                            "Exact fields use case/whitespace normalization only; amounts compare exact cents, dates use declared dataset order. No fuzzy credit.",
                            "Both versions use identical cached RustO PP-OCRv6 spans. Changes improve extraction, not handwriting recognition."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "verify-data", "freeze", "run-dev", "run-test", "report", "verify-report"))
    args = parser.parse_args()
    manifest = read(MANIFEST)
    if args.command == "download":
        fetch(manifest)
    elif args.command == "verify-data":
        verify_data(manifest)
    elif args.command == "freeze":
        if (DATA / "improved-test.jsonl").exists():
            raise ValueError("held-out evaluation already exists: cannot silently refreeze")
        write(PROFILE, {"sources": code_hashes(), "manifest_sha256": digest(MANIFEST), "models": {str(p.relative_to(ROOT)): digest(p) for p in (CRATE / "models/ppocrv6-tiny").iterdir() if p.name in ("det.rten", "rec.rten", "dict.txt")}})
        print("financial extraction frozen")
    elif args.command.startswith("run-"):
        execute(manifest, args.command.removeprefix("run-"))
    elif args.command == "report":
        write(REPORT, build_report(manifest))
        print("financial report written")
    else:
        if read(REPORT) != build_report(manifest):
            raise ValueError("stale or altered financial report")
        print("financial report verification passed")


if __name__ == "__main__":
    main()
