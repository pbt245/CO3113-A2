"""64-bit perceptual hash (pHash) and Hamming-distance search for near-duplicate detection.

pHash (same definition as the `imagehash` package):
  grayscale -> 32x32 (Lanczos) -> 2-D DCT-II -> keep the top-left 8x8 low frequencies
  -> bit = coefficient > median of those 64 coefficients.
Two images are near-duplicates when the Hamming distance of their hashes is <= a threshold
(4 bits in this project).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dctn

_BIT_WEIGHTS = (1 << np.arange(63, -1, -1, dtype=np.uint64)).astype(np.uint64)


def phash(img: Image.Image) -> int:
    small = img.convert("L").resize((32, 32), Image.LANCZOS)
    coeffs = dctn(np.asarray(small, dtype=np.float64), type=2, norm="ortho")[:8, :8]
    bits = (coeffs > np.median(coeffs)).ravel().astype(np.uint64)
    return int((bits * _BIT_WEIGHTS).sum())


def hex_to_uint64(values) -> np.ndarray:
    return np.array([int(v, 16) for v in values], dtype=np.uint64)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):          # numpy >= 2.0
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)


def near_duplicate_pairs(query: np.ndarray, ref: np.ndarray, max_dist: int = 4,
                         same_set: bool = False, chunk: int = 128):
    """All (i, j, distance) with hamming(query[i], ref[j]) <= max_dist.
    same_set=True: query and ref are the same array; only pairs i < j are returned."""
    out_i, out_j, out_d = [], [], []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        d = _popcount(np.bitwise_xor(q[:, None], ref[None, :]))
        ii, jj = np.nonzero(d <= max_dist)
        ii = ii + start
        if same_set:
            keep = ii < jj
            ii, jj = ii[keep], jj[keep]
        out_i.append(ii)
        out_j.append(jj)
        out_d.append(d[ii - start, jj])
    if not out_i:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    return (np.concatenate(out_i).astype(np.int64), np.concatenate(out_j).astype(np.int64),
            np.concatenate(out_d).astype(np.int64))
