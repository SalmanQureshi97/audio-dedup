"""Cross-dataset duplicate search using Chromaprint.

Given two fingerprint Parquet files (produced by `fingerprint_dataset.py`),
finds pairs with Chromaprint similarity above a threshold, using a
duration bucket pre-filter plus an optional MinHash coarse filter.

Usage:
    python scripts/find_duplicates.py \
        --a outputs/fma_real.parquet \
        --b outputs/sonics_real.parquet \
        --threshold 0.85 \
        --out outputs/fma_sonics_clashes.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fingerprint import (
    compare_fingerprints,
    fingerprint_from_bytes,
    minhash_signature,
)


# Chromaprint emits ~8 fingerprint ints per second of audio.
FP_INTS_PER_SECOND = 8


def parse_args():
    p = argparse.ArgumentParser(description="Find duplicate audio across two fingerprint sets.")
    p.add_argument("--a", required=True, help="Parquet A (e.g. fma_real).")
    p.add_argument("--b", required=True, help="Parquet B (e.g. sonics_real).")
    p.add_argument("--out", required=True, help="Output CSV path for matches.")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--duration-tolerance", type=float, default=0.10,
                   help="Fractional duration tolerance for the bucket pre-filter.")
    p.add_argument("--minhash-bands", type=int, default=8,
                   help="Number of MinHash bands (more = higher recall, more candidates).")
    p.add_argument("--minhash-rows", type=int, default=4,
                   help="Rows per MinHash band.")
    p.add_argument("--no-minhash", action="store_true",
                   help="Disable the MinHash coarse filter; use duration-only pre-filter.")
    return p.parse_args()


def load_fingerprints(path):
    df = pd.read_parquet(path)
    df = df[df["fingerprint"].map(lambda x: x is not None and len(x) > 0)].reset_index(drop=True)
    df["_fp_ints"] = df["fingerprint"].map(fingerprint_from_bytes)
    return df


def build_minhash_index(df, num_bands, rows_per_band):
    """Compute MinHash signatures and return band-hash -> row_idx lists."""
    num_hashes = num_bands * rows_per_band
    sigs = [minhash_signature(fp, num_hashes=num_hashes, seed=0) for fp in df["_fp_ints"]]
    band_index = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band in range(num_bands):
            start = band * rows_per_band
            band_key = (band, tuple(sig[start:start + rows_per_band]))
            band_index[band_key].append(i)
    return sigs, band_index


def candidate_pairs(df_a, df_b, args):
    """Yield (idx_a, idx_b) candidate pairs after pre-filtering."""
    # Duration bucket index on B keyed by int(seconds) with ±tolerance window.
    tol = args.duration_tolerance

    if args.no_minhash:
        # Duration-only pre-filter.
        b_by_dur = sorted((float(d), i) for i, d in enumerate(df_b["duration"]))
        durs_b = [d for d, _ in b_by_dur]
        idxs_b = [i for _, i in b_by_dur]
        import bisect
        for i_a, dur_a in enumerate(df_a["duration"]):
            lo = dur_a * (1.0 - tol)
            hi = dur_a * (1.0 + tol)
            left = bisect.bisect_left(durs_b, lo)
            right = bisect.bisect_right(durs_b, hi)
            for k in range(left, right):
                yield i_a, idxs_b[k]
        return

    # MinHash coarse filter.
    sigs_b, band_index = build_minhash_index(df_b, args.minhash_bands, args.minhash_rows)
    for i_a, (fp_a, dur_a) in enumerate(zip(df_a["_fp_ints"], df_a["duration"])):
        sig_a = minhash_signature(fp_a, num_hashes=args.minhash_bands * args.minhash_rows, seed=0)
        seen = set()
        for band in range(args.minhash_bands):
            start = band * args.minhash_rows
            key = (band, tuple(sig_a[start:start + args.minhash_rows]))
            for j in band_index.get(key, ()):
                if j in seen:
                    continue
                seen.add(j)
                dur_b = df_b["duration"].iat[j]
                if abs(dur_a - dur_b) / max(dur_a, dur_b, 1e-6) <= tol:
                    yield i_a, j


def main():
    args = parse_args()
    print(f"Loading {args.a} ...")
    df_a = load_fingerprints(args.a)
    print(f"Loading {args.b} ...")
    df_b = load_fingerprints(args.b)
    print(f"|A|={len(df_a)}  |B|={len(df_b)}  (naive pairs = {len(df_a) * len(df_b)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matches = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a_filepath", "b_filepath", "similarity", "overlap_seconds"])

        pair_iter = candidate_pairs(df_a, df_b, args)
        for i_a, i_b in tqdm(pair_iter, desc="compare"):
            fp_a = df_a["_fp_ints"].iat[i_a]
            fp_b = df_b["_fp_ints"].iat[i_b]
            sim = compare_fingerprints(fp_a, fp_b)
            if sim >= args.threshold:
                overlap_ints = min(len(fp_a), len(fp_b))
                overlap_seconds = overlap_ints / FP_INTS_PER_SECOND
                writer.writerow([
                    df_a["filepath"].iat[i_a],
                    df_b["filepath"].iat[i_b],
                    f"{sim:.4f}",
                    f"{overlap_seconds:.2f}",
                ])
                matches += 1

    print(f"Found {matches} pairs at similarity >= {args.threshold}. Wrote {out_path}.")


if __name__ == "__main__":
    main()
