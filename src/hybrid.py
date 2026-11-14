"""Hybrid scenes: real NISAR GUNW background + injected landslides with known outlines.

Real GUNW pairs carry real atmosphere, ionosphere residue, snow and field decorrelation,
but no landslide labels. Purely synthetic scenes have labels but idealised noise. We take
the real pair as the background and add landslide signatures (downslope ellipse, LOS
subsidence, coherence drop) drawn from the same rules as the synthetic generator, so the
label is exact and everything else in the scene is real.

Split is by date, not by pixel: the earliest pairs feed training, the last pairs are held
out, so the test background was never seen during training.

Caveat: if a real landslide moved in a held-out pair, it is unlabelled and a correct
detection of it counts as a false alarm — the scores here are a lower bound.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .physics import SARPhysics

N_TEST_PAIRS = 2


def consecutive_pairs(items):
    """(previous pair, pair) where previous secondary date == this reference date."""
    out = []
    for it in items:
        prev = next((p for p in items if p["sec_date"] == it["ref_date"]), None)
        if prev is not None:
            out.append((prev, it))
    out.sort(key=lambda pc: pc[1]["sec_date"])
    return out


def split_pairs(items, n_test=N_TEST_PAIRS):
    pairs = consecutive_pairs(items)
    if len(pairs) <= n_test:
        raise RuntimeError(f"Need more than {n_test} consecutive real pairs, have {len(pairs)}. "
                           "Run scripts/fetch_almaty.py first.")
    return pairs[:-n_test], pairs[-n_test:]


def proxy_amplitude(shape, dem):
    """GUNW has no amplitude layer; same terrain backscatter model as the app uses."""
    rng = np.random.default_rng(0)
    _, slope, gx, _ = SARPhysics._terrain(shape, rng, dem)
    return SARPhysics._amplitude(slope, gx, rng, shape).astype(np.float32)


def hybrid_scene(prev, cur, dem, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    valid = cur["valid"].astype(bool) & prev["valid"].astype(bool)
    coh_before = np.where(valid, prev["coherence"], 0).astype(np.float64)
    coh_after = np.where(valid, cur["coherence"], 0).astype(np.float64)
    phase = np.where(valid, cur["unwrapped_phase"], 0).astype(np.float64)
    wavelength = float(cur["wavelength"])
    shape = phase.shape
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dem_, slope, gx, gy = SARPhysics._terrain(shape, rng, dem)

    # Same placement rule as the synthetic generator, plus: only where the real pair is valid
    # and was coherent before (a slide on an already-decorrelated slope is invisible anyway)
    weight = (np.clip((slope - 12) / 10, 0, 1) * np.clip((40 - slope) / 8, 0, 1)
              * np.clip((2700 - dem_) / 400, 0, 1)
              * (valid & (coh_before > 0.35))) + 1e-6
    truth = np.zeros(shape, bool)
    anomalies = []
    for cy, cx in SARPhysics._pick_centres(weight, int(rng.integers(3, 7)), rng,
                                           0.14 * min(h, w)):
        sy, sx = rng.uniform(3.5, 9), rng.uniform(5, 13)
        theta = float(np.arctan2(gy[int(cy), int(cx)], gx[int(cy), int(cx)]))
        g = SARPhysics._ellipse(xx, yy, cy, cx, sy, sx, theta)
        kind = "rapid" if rng.random() < 0.55 else "slow-coherent"
        disp_mm = -rng.uniform(25, 110) if kind == "rapid" else -rng.uniform(20, 60)
        drop = rng.uniform(0.32, 0.55) if kind == "rapid" else rng.uniform(0.05, 0.16)
        phase += SARPhysics.displacement_to_phase(disp_mm, wavelength) * g
        new_coh = np.clip(coh_after - drop * g, 0.02, 1)
        # Extra phase noise only where we removed coherence
        extra = np.sqrt(np.clip(1 - new_coh ** 2 - (1 - coh_after ** 2), 0, 1))
        phase += rng.normal(0, 1, shape) * gaussian_filter(extra, 1) * 0.6
        coh_after = new_coh
        truth |= g > 0.35
        anomalies.append(dict(cy=cy, cx=cx, kind=kind, disp_mm=round(float(disp_mm), 1)))

    truth &= valid
    phase = np.where(valid, phase, 0)
    coh_after = np.where(valid, coh_after, 0)
    return {
        "unwrapped_phase": phase.astype(np.float32),
        "coherence_before": coh_before.astype(np.float32),
        "coherence_after": coh_after.astype(np.float32),
        "amplitude": proxy_amplitude(shape, dem),
        "truth_mask": truth, "valid": valid, "wavelength": wavelength,
        "anomalies": anomalies, "seed": seed,
        "pair": f"{cur['ref_date']} → {cur['sec_date']}",
    }


def hybrid_scenes(pairs, dem, seeds):
    for prev, cur in pairs:
        for s in seeds:
            yield hybrid_scene(prev, cur, dem, s)
