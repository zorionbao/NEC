#!/usr/bin/env python3
"""Stream GEO Bismark coverage files into sample-level 1-kb methylation bins."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "data" / "public" / "eclipse_nec"
RAW_COV_ROOT = PUBLIC_ROOT / "raw_cov"
PROCESSED_ROOT = ROOT / "data" / "processed" / "eclipse_nec"
BIN_ROOT = PROCESSED_ROOT / "bin_1kb"
MANIFESTS = ROOT / "manifests"
METADATA_PATH = PUBLIC_ROOT / "eclipse_sample_metadata.csv"
BIN_SIZE = 1_000
CHUNK_ROWS = 500_000
STANDARD_CHROMOSOMES = {f"chr{value}" for value in range(1, 23)} | {
    "chrX",
    "chrY",
}
GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_46/gencode.v46.basic.annotation.gtf.gz"
)
GTF_PATH = PUBLIC_ROOT / "gencode.v46.basic.annotation.gtf.gz"


def ensure_local(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside local analysis root: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_gtf() -> None:
    if GTF_PATH.exists() and GTF_PATH.stat().st_size > 25_000_000:
        return
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for the pinned GENCODE download")
    command = [
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
        "--continue-at",
        "-",
        "--output",
        str(ensure_local(GTF_PATH)),
        GTF_URL,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"GENCODE download failed with code {completed.returncode}"
        )


def normalize_chromosome(values: pd.Series) -> pd.Series:
    normalized = values.astype(str)
    has_prefix = normalized.str.startswith("chr")
    normalized = normalized.where(has_prefix, "chr" + normalized)
    return normalized


def aggregate_sample(
    input_path: Path,
    gsm: str,
    accession: str,
) -> dict[str, object]:
    output = ensure_local(BIN_ROOT / f"{gsm}_bin_1kb.parquet")
    if output.exists() and output.stat().st_size > 10_000:
        existing = pd.read_parquet(
            output, columns=["chromosome", "bin_start", "cpgs"]
        )
        return {
            "accession": accession,
            "gsm": gsm,
            "status": "REUSED",
            "cpg_rows": int(existing["cpgs"].sum()),
            "bins": int(existing.shape[0]),
            "output": str(output.relative_to(ROOT)),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        }

    partials: list[pd.DataFrame] = []
    cpg_rows = 0
    with gzip.open(input_path, mode="rb") as decompressed:
        reader = pd.read_csv(
            decompressed,
            sep="\t",
            header=None,
            usecols=[0, 1, 4, 5],
            names=["chromosome", "position", "methylated", "unmethylated"],
            dtype={
                "chromosome": "string",
                "position": "int64",
                "methylated": "int64",
                "unmethylated": "int64",
            },
            chunksize=CHUNK_ROWS,
            on_bad_lines="skip",
        )
        for chunk in reader:
            chunk["chromosome"] = normalize_chromosome(chunk["chromosome"])
            chunk = chunk[chunk["chromosome"].isin(STANDARD_CHROMOSOMES)].copy()
            if chunk.empty:
                continue
            cpg_rows += int(len(chunk))
            position = chunk["position"].to_numpy(dtype=np.int64)
            if np.any(position < 1):
                raise RuntimeError(
                    f"{gsm}: Bismark COV contains non-positive CpG positions"
                )
            # Bismark COV positions are 1-based; GENCODE intervals below are
            # represented as 0-based half-open coordinates.
            chunk["bin_start"] = ((position - 1) // BIN_SIZE) * BIN_SIZE
            grouped = (
                chunk.groupby(
                    ["chromosome", "bin_start"],
                    observed=True,
                    sort=False,
                )
                .agg(
                    methylated=("methylated", "sum"),
                    unmethylated=("unmethylated", "sum"),
                    cpgs=("position", "size"),
                )
                .reset_index()
            )
            partials.append(grouped)
    if not partials:
        raise RuntimeError(f"{gsm}: no standard-chromosome CpGs were parsed")
    bins = (
        pd.concat(partials, ignore_index=True)
        .groupby(["chromosome", "bin_start"], observed=True, sort=True)
        .agg(
            methylated=("methylated", "sum"),
            unmethylated=("unmethylated", "sum"),
            cpgs=("cpgs", "sum"),
        )
        .reset_index()
    )
    bins["coverage"] = bins["methylated"] + bins["unmethylated"]
    bins["beta"] = np.divide(
        bins["methylated"].to_numpy(dtype=float),
        bins["coverage"].to_numpy(dtype=float),
        out=np.full(len(bins), np.nan, dtype=float),
        where=bins["coverage"].to_numpy(dtype=float) > 0,
    ).astype(np.float32)
    bins["bin_end"] = bins["bin_start"] + BIN_SIZE
    bins = bins[
        [
            "chromosome",
            "bin_start",
            "bin_end",
            "methylated",
            "unmethylated",
            "coverage",
            "cpgs",
            "beta",
        ]
    ]
    bins.to_parquet(output, index=False, compression="zstd")
    return {
        "accession": accession,
        "gsm": gsm,
        "status": "PROCESSED",
        "cpg_rows": cpg_rows,
        "bins": int(len(bins)),
        "output": str(output.relative_to(ROOT)),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def process_sample(
    root_text: str,
    accession: str,
    gsm: str,
    filename: str,
) -> dict[str, object]:
    root = Path(root_text)
    input_path = (
        root / "data" / "public" / "eclipse_nec" / "raw_cov" / filename
    )
    if not input_path.exists():
        raise RuntimeError(f"{gsm}: validated COV input is missing")
    start = time.time()
    row = aggregate_sample(input_path, gsm, accession)
    row["elapsed_seconds"] = time.time() - start
    print(
        f"BIN_COMPLETE {accession} {gsm} {row['bins']}",
        flush=True,
    )
    return row


def parse_gtf() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(GTF_PATH, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chromosome = fields[0]
            if not chromosome.startswith("chr"):
                chromosome = "chr" + chromosome
            if chromosome not in STANDARD_CHROMOSOMES:
                continue
            attributes: dict[str, str] = {}
            for item in fields[8].split(";"):
                item = item.strip()
                if not item or " " not in item:
                    continue
                key, value = item.split(" ", 1)
                attributes[key] = value.strip().strip('"')
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            transcription_start = start if strand == "+" else end - 1
            rows.append(
                {
                    "chromosome": chromosome,
                    "gene_start": start,
                    "gene_end": end,
                    "strand": strand,
                    "transcription_start": transcription_start,
                    "promoter_start": max(0, transcription_start - 2_000),
                    "promoter_end": transcription_start + 2_001,
                    "gene_id": attributes.get("gene_id", "").split(".")[0],
                    "gene": attributes.get("gene_name", ""),
                    "gene_type": attributes.get("gene_type", ""),
                }
            )
    genes = pd.DataFrame(rows)
    genes = genes[genes["gene"].ne("")].drop_duplicates(
        ["chromosome", "gene_start", "gene_end", "gene"]
    )
    output = ensure_local(PROCESSED_ROOT / "gencode_v46_gene_promoters.parquet")
    genes.to_parquet(output, index=False, compression="zstd")
    return genes


def main() -> None:
    ensure_local(BIN_ROOT).mkdir(parents=True, exist_ok=True)
    ensure_local(PROCESSED_ROOT).mkdir(parents=True, exist_ok=True)
    download_manifest = json.loads(
        (MANIFESTS / "eclipse_download_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if download_manifest["status"] != "PASS":
        raise RuntimeError("ECLIPSE download manifest is not PASS")
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    summaries: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_lookup = {
            executor.submit(
                process_sample,
                str(ROOT),
                str(row.accession),
                str(row.gsm),
                str(row.supplementary_filename),
            ): str(row.gsm)
            for row in metadata.itertuples(index=False)
        }
        for future in as_completed(future_lookup):
            summaries.append(future.result())

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["accession", "gsm"]
    )
    summary_path = ensure_local(PROCESSED_ROOT / "eclipse_bin_preparation.csv")
    summary_frame.to_csv(summary_path, index=False)
    if summary_frame["gsm"].nunique() != metadata["gsm"].nunique():
        raise RuntimeError("Not all GEO samples produced 1-kb bin outputs")
    if (summary_frame["bins"] < 10_000).any():
        raise RuntimeError("At least one sample produced too few methylation bins")

    download_gtf()
    genes = parse_gtf()
    manifest = {
        "status": "PASS",
        "analysis": "ECLIPSE-NEC 1-kb methylation preparation",
        "bin_size_bp": BIN_SIZE,
        "coverage_source": "full processed Bismark COV rows for every public sample",
        "samples": int(summary_frame["gsm"].nunique()),
        "accessions": sorted(summary_frame["accession"].unique().tolist()),
        "total_cpg_rows": int(summary_frame["cpg_rows"].fillna(0).sum()),
        "median_bins_per_sample": float(summary_frame["bins"].median()),
        "gencode_release": 46,
        "genes": int(genes["gene"].nunique()),
        "gtf_path": str(GTF_PATH.relative_to(ROOT)),
        "gtf_sha256": sha256(GTF_PATH),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path),
        "cloud_files_written": 0,
    }
    (MANIFESTS / "eclipse_bin_preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
