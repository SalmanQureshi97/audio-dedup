"""Shared Chromaprint utilities.

Wraps `pyacoustid` so the rest of the package can stay agnostic of the
underlying C library. Fingerprints are stored as lists of 32-bit unsigned
integers; helpers convert to/from bytes for Parquet storage.
"""

from __future__ import annotations

import struct
from typing import List, Tuple

import acoustid
import chromaprint


def compute_fingerprint(path: str) -> Tuple[float, List[int]]:
    """Compute a chromaprint fingerprint for a single audio file.

    Returns (duration_seconds, fingerprint_uint32_list).
    Raises `acoustid.FingerprintGenerationError` if fpcalc fails.
    """
    duration, fp_str = acoustid.fingerprint_file(path)
    # `fp_str` is the base64-ish encoded blob fpcalc prints. Decode to the
    # raw uint32 stream that chromaprint.compare_fingerprints expects.
    fp_ints, _version = chromaprint.decode_fingerprint(
        fp_str.encode("ascii") if isinstance(fp_str, str) else fp_str
    )
    return float(duration), list(fp_ints)


def compare_fingerprints(fp_a: List[int], fp_b: List[int]) -> float:
    """Return similarity score in [0, 1].

    Uses chromaprint's window-aligned comparison (handles different
    durations by finding the best-matching overlap region). Falls back to
    a manual bit-error-rate over the aligned region if the binding does
    not expose `compare_fingerprints` directly.
    """
    if hasattr(chromaprint, "compare_fingerprints"):
        return float(chromaprint.compare_fingerprints(fp_a, fp_b))
    return _manual_similarity(fp_a, fp_b)


def _manual_similarity(fp_a: List[int], fp_b: List[int]) -> float:
    """Sliding-window bit-error-rate similarity fallback.

    Slides the shorter fingerprint over the longer one and returns the
    best (1 - normalised_hamming) score. O(len_a * len_b * 32) — fine for
    fingerprints in the low thousands of uint32s (a few minutes of audio).
    """
    if not fp_a or not fp_b:
        return 0.0
    if len(fp_a) > len(fp_b):
        fp_a, fp_b = fp_b, fp_a
    n = len(fp_a)
    best_sim = 0.0
    for offset in range(len(fp_b) - n + 1):
        hamming = 0
        for x, y in zip(fp_a, fp_b[offset:offset + n]):
            hamming += bin((x ^ y) & 0xFFFFFFFF).count("1")
        sim = 1.0 - hamming / (n * 32.0)
        if sim > best_sim:
            best_sim = sim
    return best_sim


def fingerprint_to_bytes(fp: List[int]) -> bytes:
    """Pack a uint32 fingerprint list into a compact little-endian byte blob."""
    return struct.pack(f"<{len(fp)}I", *fp)


def fingerprint_from_bytes(blob: bytes) -> List[int]:
    """Inverse of `fingerprint_to_bytes`."""
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}I", blob[: n * 4]))


def minhash_signature(fp: List[int], num_hashes: int = 32, seed: int = 0) -> List[int]:
    """Cheap 32-dim MinHash signature over fingerprint integers.

    Used as a coarse candidate filter before exact comparison.
    """
    import numpy as np

    if not fp:
        return [0] * num_hashes
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31 - 1, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, 2**31 - 1, size=num_hashes, dtype=np.int64)
    prime = (1 << 31) - 1
    arr = np.asarray(fp, dtype=np.int64)
    # shape: (num_hashes, len(fp))
    hashed = (a[:, None] * arr[None, :] + b[:, None]) % prime
    return hashed.min(axis=1).tolist()
