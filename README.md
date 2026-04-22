# audio-dedup

Length-invariant audio fingerprinting and cross-dataset duplicate detection
for the TTM-detector thesis. Verifies the provenance-only claim that FMA
real and SONICS real do not share underlying recordings.

## Why Chromaprint

- **Length-invariant.** Chromaprint is a sliding-window chroma fingerprint;
  `chromaprint.compare_fingerprints` aligns windows, so a 30-second clip and
  a 5-minute track that share a segment still match.
- **Format/bitrate robust.** Computed on downsampled chroma features; survives
  MP3 ↔ WAV re-encoding and bitrate changes.
- **Fast.** ~0.1 s per track on CPU, trivially parallelisable.
- **Battle-tested.** The backbone of MusicBrainz / AcoustID.

## Install

```bash
pip install -r requirements.txt

# System binary (pyacoustid shells out to fpcalc):
sudo apt-get install -y libchromaprint-tools   # Debian/Ubuntu
brew install chromaprint                        # macOS
```

## Usage

### 1. Fingerprint a dataset

```bash
python scripts/fingerprint_dataset.py \
    --dir /home/jovyan/Thesis/Code/data/fma_real \
    --out outputs/fma_real.parquet \
    --workers 8
```

Writes a Parquet file with columns `[filepath, track_id, duration, fingerprint]`.
Resumable — rows already present in the output Parquet are skipped on re-run.

### 2. Find cross-dataset duplicates

```bash
python scripts/find_duplicates.py \
    --a outputs/fma_real.parquet \
    --b outputs/sonics_real.parquet \
    --threshold 0.85 \
    --out outputs/fma_sonics_clashes.csv
```

Strategy:
1. Duration bucket pre-filter (±10%) cuts the pairwise candidate count.
2. 32-bit MinHash over chromaprint integers gives a sublinear coarse
   candidate retrieval when either side is large (>10 k tracks).
3. Exact `chromaprint.compare_fingerprints` on surviving candidates.

Output CSV: `[a_filepath, b_filepath, similarity, overlap_seconds]` for all
pairs at or above the threshold.

## Interpreting similarity scores

| Score        | Meaning                                                    |
|--------------|------------------------------------------------------------|
| ≥ 0.95       | Near-identical — same recording, same master.              |
| 0.85 – 0.95  | Likely the same song, different release / master / edit.   |
| < 0.85       | Different content; coincidental feature overlap.           |

Default threshold: **0.85**. Use `--threshold 0.90` for a stricter sweep.

## Handling confirmed duplicates

Downstream in TTM-detector, any pair flagged at or above the chosen threshold
should either:

1. Be aliased to a shared `track_id` so the group-aware split keeps them in
   the same partition, or
2. Be dropped from the manifest before split assignment.

The SONICS-vs-FMA run is expected to return an empty match list given the
datasets' disjoint provenance. An empty result empirically backs the
provenance argument used in the thesis.
