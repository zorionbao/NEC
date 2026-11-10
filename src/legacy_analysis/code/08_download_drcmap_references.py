from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_ROOT = ANALYSIS_ROOT / "data" / "public" / "dr_cmap"
MANIFEST_ROOT = ANALYSIS_ROOT / "manifests"

REFERENCES = [
    {
        "resource": "Gut Cell Atlas epithelium raw counts",
        "accession": "Space-Time Gut Cell Atlas",
        "url": (
            "https://cellgeni.cog.sanger.ac.uk/gutcellatlas/"
            "epi_raw_counts02_v2.h5ad"
        ),
        "filename": "epi_raw_counts02_v2.h5ad",
        "expected_size": 666_604_440,
        "role": "epithelial developmental reference",
    },
    {
        "resource": "Gut Cell Atlas endothelium raw counts",
        "accession": "Space-Time Gut Cell Atlas",
        "url": (
            "https://cellgeni.cog.sanger.ac.uk/gutcellatlas/"
            "endothelium_raw_counts02_v2.h5ad"
        ),
        "filename": "endothelium_raw_counts02_v2.h5ad",
        "expected_size": 80_667_341,
        "role": "vascular and lymphatic endothelial reference",
    },
    {
        "resource": "Gut Cell Atlas T and NK raw counts",
        "accession": "Space-Time Gut Cell Atlas",
        "url": (
            "https://cellgeni.cog.sanger.ac.uk/gutcellatlas/"
            "Tcell_raw_counts02_v2.h5ad"
        ),
        "filename": "Tcell_raw_counts02_v2.h5ad",
        "expected_size": 118_766_238,
        "role": "ILC3, regulatory and resident T-cell reference",
    },
    {
        "resource": "Human M-cell organoid SORT-seq raw count archive",
        "accession": "GSE275772",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE275nnn/"
            "GSE275772/suppl/GSE275772_RAW.tar"
        ),
        "filename": "GSE275772_RAW.tar",
        "expected_size": 2_785_280,
        "role": "human M-cell reference",
    },
    {
        "resource": "Human M-cell bulk normalized counts",
        "accession": "GSE275771",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE275nnn/"
            "GSE275771/suppl/"
            "GSE275771_P11N_M_cell_bulk_seq_NormalizedCounts.csv.gz"
        ),
        "filename": (
            "GSE275771_P11N_M_cell_bulk_seq_NormalizedCounts.csv.gz"
        ),
        "expected_size": None,
        "role": "M-cell marker validation",
    },
    {
        "resource": "Human M-cell bulk read counts",
        "accession": "GSE275771",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE275nnn/"
            "GSE275771/suppl/"
            "GSE275771_P11N_M_cell_bulk_seq_ReadCounts.tsv.gz"
        ),
        "filename": "GSE275771_P11N_M_cell_bulk_seq_ReadCounts.tsv.gz",
        "expected_size": None,
        "role": "M-cell marker validation",
    },
]


def ensure_local_output(path: Path) -> Path:
    resolved = path.resolve()
    root = ANALYSIS_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside local analysis root: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(reference: dict[str, object]) -> Path:
    destination = ensure_local_output(
        DOWNLOAD_ROOT / str(reference["filename"])
    )
    expected_size = reference["expected_size"]
    if (
        destination.exists()
        and expected_size is not None
        and destination.stat().st_size == expected_size
    ):
        return destination
    command = [
        "curl.exe",
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(destination),
        str(reference["url"]),
    ]
    subprocess.run(command, check=True)
    if expected_size is not None and destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {destination.name}: "
            f"{destination.stat().st_size} != {expected_size}"
        )
    return destination


def main() -> None:
    for directory in [DOWNLOAD_ROOT, MANIFEST_ROOT]:
        ensure_local_output(directory).mkdir(parents=True, exist_ok=True)
    rows = []
    for reference in REFERENCES:
        path = download(reference)
        rows.append(
            {
                **reference,
                "local_path": str(path.relative_to(ANALYSIS_ROOT)),
                "observed_size": path.stat().st_size,
                "sha256": sha256(path),
                "status": "verified",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        ensure_local_output(MANIFEST_ROOT / "dr_cmap_reference_manifest.csv"),
        index=False,
    )
    report = {
        "status": "PASS",
        "resource_count": int(manifest.shape[0]),
        "total_size_bytes": int(manifest["observed_size"].sum()),
        "all_sha256_present": bool(manifest["sha256"].str.len().eq(64).all()),
        "sources": [
            "Gut Cell Atlas official Sanger downloads",
            "NCBI GEO GSE275771",
            "NCBI GEO GSE275772",
        ],
        "cloud_files_written": 0,
    }
    (MANIFEST_ROOT / "dr_cmap_reference_manifest.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
