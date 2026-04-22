"""Walk a directory of audio files and write their Chromaprint fingerprints to Parquet.

Usage:
    python scripts/fingerprint_dataset.py \
        --dir /home/jovyan/Thesis/Code/data/fma_real \
        --out outputs/fma_real.parquet \
        --workers 8
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fingerprint import compute_fingerprint, fingerprint_to_bytes


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args():
    p = argparse.ArgumentParser(description="Fingerprint a directory of audio files.")
    p.add_argument("--dir", required=True, help="Input audio directory (searched recursively).")
    p.add_argument("--out", required=True, help="Output Parquet path.")
    p.add_argument("--workers", type=int, default=4, help="Worker processes.")
    return p.parse_args()


def _worker(path_str: str):
    try:
        duration, fp = compute_fingerprint(path_str)
        return path_str, duration, fingerprint_to_bytes(fp), None
    except Exception as err:
        return path_str, None, None, f"{type(err).__name__}: {err}"


def collect_files(root: Path):
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            if ".ipynb_checkpoints" in p.parts:
                continue
            files.append(str(p.resolve()))
    return files


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_files = collect_files(Path(args.dir))
    print(f"Discovered {len(all_files)} audio files under {args.dir}")

    # Resume: skip files already fingerprinted in the output Parquet.
    existing = pd.DataFrame(columns=["filepath", "track_id", "duration", "fingerprint"])
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(existing["filepath"])
        before = len(all_files)
        all_files = [f for f in all_files if f not in done]
        print(f"Resuming: {before - len(all_files)} already fingerprinted, {len(all_files)} remaining.")

    rows = []
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, f): f for f in all_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fingerprint"):
            path_str, duration, fp_bytes, err = fut.result()
            if err is not None:
                errors += 1
                continue
            rows.append({
                "filepath": path_str,
                "track_id": Path(path_str).stem,
                "duration": duration,
                "fingerprint": fp_bytes,
            })

    new_df = pd.DataFrame(rows)
    out_df = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path} ({errors} errors).")


if __name__ == "__main__":
    main()
