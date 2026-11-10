#!/usr/bin/env python3
"""Download and audit public NEC methylation coverage data from GEO."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "data" / "public" / "eclipse_nec"
RAW_COV_ROOT = PUBLIC_ROOT / "raw_cov"
MANIFESTS = ROOT / "manifests"
PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
RAW_COV_ROOT.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

ACCESSIONS = {
    "GSE212913": {
        "cohort": "ileum",
        "expected_samples": 15,
    },
    "GSE212997": {
        "cohort": "colon_stool",
        "expected_samples": 37,
    },
    "GSE213704": {
        "cohort": "blood",
        "expected_samples": 28,
    },
}


def ensure_local(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside local analysis root: {resolved}")
    return resolved


def fetch_text(url: str) -> str:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for GEO metadata downloads")
    completed = subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "4",
            "--user-agent",
            "NEC-ECLIPSE-public-data-audit/1.0",
            url,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GEO metadata download failed: {completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout.decode("utf-8", errors="replace")


def parse_soft(accession: str, text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block in text.split("^SAMPLE = ")[1:]:
        lines = block.splitlines()
        gsm = lines[0].strip()
        title = next(
            (
                line.split(" = ", 1)[1]
                for line in lines
                if line.startswith("!Sample_title = ")
            ),
            "",
        )
        characteristics: dict[str, str] = {}
        for line in lines:
            if not line.startswith("!Sample_characteristics_ch1 = "):
                continue
            value = line.split(" = ", 1)[1]
            if ": " in value:
                key, item = value.split(": ", 1)
                characteristics[key.strip().lower()] = item.strip()
        supplementary = next(
            (
                line.split(" = ", 1)[1]
                for line in lines
                if line.startswith("!Sample_supplementary_file_1 = ")
            ),
            "",
        )
        title_parts = [value.strip() for value in title.split(",")]
        subject = characteristics.get("patient id", "")
        if not subject and len(title_parts) >= 2:
            subject = title_parts[1]
        tissue = characteristics.get("tissue", "")
        disease = characteristics.get("disease state", "")
        normalized = {
            "NEC": "nec",
            "non-NEC": "non_nec",
            "control": "control",
            "control turns to NEC": "future_nec",
        }.get(disease, disease.lower().replace(" ", "_"))
        rows.append(
            {
                "accession": accession,
                "gsm": gsm,
                "title": title,
                "subject_id": subject,
                "tissue": tissue,
                "disease_state": disease,
                "analysis_condition": normalized,
                "race": characteristics.get("race", ""),
                "sex": characteristics.get("sex", ""),
                "corrected_gestational_age_at_sampling": characteristics.get(
                    "cga (corrected_gestational_age)at_sampling", ""
                ),
                "days_of_life_at_sampling": characteristics.get(
                    "dol (days_of_life)_at_sampling", ""
                ),
                "days_of_life_at_nec_onset": characteristics.get(
                    "dol at_nec_onset", ""
                ),
                "supplementary_url": supplementary.replace(
                    "ftp://ftp.ncbi.nlm.nih.gov",
                    "https://ftp.ncbi.nlm.nih.gov",
                ),
                "supplementary_filename": supplementary.rsplit("/", 1)[-1],
            }
        )
    return rows


def validate_gzip(path: Path) -> int:
    decompressed_bytes = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            decompressed_bytes += len(block)
    if decompressed_bytes <= 0:
        raise RuntimeError(f"Empty gzip payload: {path.name}")
    return decompressed_bytes


def download_sample(row: dict[str, object]) -> dict[str, object]:
    accession = str(row["accession"])
    gsm = str(row["gsm"])
    filename = str(row["supplementary_filename"])
    target = ensure_local(RAW_COV_ROOT / filename)
    if target.exists() and target.stat().st_size > 1_000_000:
        try:
            decompressed_bytes = validate_gzip(target)
            return {
                "accession": accession,
                "gsm": gsm,
                "filename": filename,
                "status": "REUSED",
                "bytes": target.stat().st_size,
                "decompressed_bytes": decompressed_bytes,
                "sha256": sha256(target),
            }
        except (OSError, EOFError):
            target.unlink()
    temporary = ensure_local(target.with_suffix(target.suffix + ".part"))
    temporary.unlink(missing_ok=True)
    url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        f"?acc={quote(gsm)}&format=file&file={quote(filename)}"
    )
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for GEO downloads")
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "12",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--output",
        str(temporary),
        url,
    ]
    print(f"DOWNLOAD_START {accession} {gsm} {filename}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"curl failed for {gsm} with code {completed.returncode}"
        )
    decompressed_bytes = validate_gzip(temporary)
    os.replace(temporary, target)
    print(
        f"DOWNLOAD_COMPLETE {accession} {gsm} {target.stat().st_size}",
        flush=True,
    )
    return {
        "accession": accession,
        "gsm": gsm,
        "filename": filename,
        "status": "DOWNLOADED",
        "bytes": target.stat().st_size,
        "decompressed_bytes": decompressed_bytes,
        "sha256": sha256(target),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    metadata_rows: list[dict[str, object]] = []
    for accession, specification in ACCESSIONS.items():
        series_url = (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
            f"targ=self&acc={accession}&form=text&view=quick"
        )
        sample_url = (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
            f"targ=gsm&acc={accession}&form=text&view=full"
        )
        series_path = ensure_local(
            PUBLIC_ROOT / f"{accession}_series.soft"
        )
        sample_path = ensure_local(
            PUBLIC_ROOT / f"{accession}_samples.soft"
        )
        if series_path.exists() and series_path.stat().st_size > 1_000:
            series_text = series_path.read_text(encoding="utf-8")
        else:
            series_text = fetch_text(series_url)
            series_path.write_text(series_text, encoding="utf-8")
        if sample_path.exists() and sample_path.stat().st_size > 10_000:
            sample_text = sample_path.read_text(encoding="utf-8")
        else:
            sample_text = fetch_text(sample_url)
            sample_path.write_text(sample_text, encoding="utf-8")
        parsed = parse_soft(accession, sample_text)
        if len(parsed) != specification["expected_samples"]:
            raise RuntimeError(
                f"{accession}: expected {specification['expected_samples']} "
                f"samples, parsed {len(parsed)}"
            )
        metadata_rows.extend(parsed)

    metadata = pd.DataFrame(metadata_rows)
    metadata_path = ensure_local(PUBLIC_ROOT / "eclipse_sample_metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    completed_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_lookup = {
            executor.submit(download_sample, row): str(row["gsm"])
            for row in metadata.to_dict(orient="records")
        }
        for future in as_completed(future_lookup):
            completed_rows.append(future.result())

    inventory = pd.DataFrame(completed_rows).sort_values(["accession", "gsm"])
    inventory_path = ensure_local(
        PUBLIC_ROOT / "eclipse_raw_cov_inventory.csv"
    )
    inventory.to_csv(inventory_path, index=False)
    if inventory["gsm"].nunique() != metadata["gsm"].nunique():
        raise RuntimeError("Not every metadata sample has a validated COV file")
    if (inventory["bytes"] <= 1_000_000).any():
        raise RuntimeError("At least one validated COV file is unexpectedly small")

    design = (
        metadata.groupby(
            ["accession", "tissue", "analysis_condition"],
            dropna=False,
            observed=True,
        )
        .agg(samples=("gsm", "nunique"), subjects=("subject_id", "nunique"))
        .reset_index()
    )
    design_path = ensure_local(PUBLIC_ROOT / "eclipse_cohort_design.csv")
    design.to_csv(design_path, index=False)

    manifest = {
        "status": "PASS",
        "analysis": "ECLIPSE-NEC public methylation download",
        "source": "NCBI GEO public processed Bismark coverage files",
        "accessions": ACCESSIONS,
        "samples": int(metadata["gsm"].nunique()),
        "subjects": int(metadata["subject_id"].nunique()),
        "tissues": sorted(metadata["tissue"].unique().tolist()),
        "download_unit": "sample-level Bismark COV gzip",
        "sample_files": int(inventory["gsm"].nunique()),
        "compressed_bytes": int(inventory["bytes"].sum()),
        "decompressed_bytes": int(inventory["decompressed_bytes"].sum()),
        "metadata_sha256": sha256(metadata_path),
        "inventory_sha256": sha256(inventory_path),
        "design_sha256": sha256(design_path),
        "cloud_files_written": 0,
    }
    (MANIFESTS / "eclipse_download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
