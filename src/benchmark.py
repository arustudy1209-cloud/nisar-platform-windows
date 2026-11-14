"""U-Net vs. a plain coherence-loss threshold, on the same held-out scenes.

Protocol (fixed before looking at results):
  * train scenes: seeds 0..N_TRAIN-1, test scenes: seeds 1000.. — never overlap
  * the threshold baseline gets two chances: the textbook value (Δγ > 0.3) and the value
    that maximises F1 on the *training* scenes, so it is not a strawman
  * both methods share the same post-processing (opening, hole filling, min area)
  * pixel metrics (precision / recall / F1) and object metrics (landslides found,
    false alarms), because for an early-warning service a missed slope matters more
    than a few boundary pixels
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from skimage import measure
from sklearn.metrics import f1_score, precision_score, recall_score

from .coherence import CoherenceAnalyzer
from .features import N_CHANNELS, scene_input
from .inference import DisasterDetector
from .model import ModelTrainer, UNet
from .physics import NISAR_L_WAVELENGTH, SARPhysics

TRAIN_SEEDS = list(range(0, 24))
TEST_SEEDS = list(range(1000, 1020))
HYBRID_TRAIN_SEEDS = list(range(2000, 2004))   # landslide layouts per real train pair
HYBRID_TEST_SEEDS = list(range(3000, 3010))    # per held-out real pair
DEFAULT_THR = 0.30
MIN_AREA = 50
INC_ANGLE = 35.0


def _wl(scene):
    return float(scene.get("wavelength", NISAR_L_WAVELENGTH))


def _valid(scene, pred):
    v = scene.get("valid")
    return pred if v is None else pred & v


def threshold_mask(scene, thr, min_area=MIN_AREA):
    dcoh = CoherenceAnalyzer.compute_coherence_change(scene["coherence_before"],
                                                      scene["coherence_after"])
    return _valid(scene, DisasterDetector.postprocess(dcoh > thr, min_area_px=min_area))


def object_metrics(truth, pred):
    """Landslides hit (any overlap) and predicted blobs with no true landslide under them."""
    t_lab = measure.label(truth, connectivity=2)
    p_lab = measure.label(pred, connectivity=2)
    n_true = int(t_lab.max())
    hit = len(set(np.unique(t_lab[pred])) - {0})
    false_alarms = len(set(range(1, int(p_lab.max()) + 1)) - set(np.unique(p_lab[truth])))
    return n_true, hit, false_alarms


def pixel_metrics(truth, pred):
    y, p = truth.ravel(), pred.ravel()
    return dict(precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0),
                f1=f1_score(y, p, zero_division=0))


def synthetic_scenes(dem, seeds):
    for s in seeds:
        yield SARPhysics.generate_synthetic_gunw(seed=s, dem=dem)


def train_unet(dem=None, seeds=TRAIN_SEEDS, epochs=25, lr=1e-3, progress=None,
               streamlit_log=False, scenes=None):
    """Train on `scenes` (any iterable of labelled scene dicts) or synthetic `seeds`."""
    scenes = synthetic_scenes(dem, seeds) if scenes is None else scenes
    trainer = ModelTrainer(UNet(in_channels=N_CHANNELS))
    for i, sc in enumerate(scenes):
        x, vert = scene_input(sc, _wl(sc), INC_ANGLE)
        trainer.prepare_tiles(x, vert, tile_size=256, overlap=64,
                              labels=sc["truth_mask"].astype(np.float32), append=i > 0)
    trainer.train(epochs=epochs, lr=lr, batch_size=8, streamlit_log=streamlit_log,
                  progress_callback=progress)
    return trainer


def displacement_mask(scene, thr_mm, min_area=MIN_AREA):
    _, vert = scene_input(scene, _wl(scene), INC_ANGLE)
    return _valid(scene, DisasterDetector.postprocess(np.abs(SARPhysics.highpass(vert)) > thr_mm,
                                                      min_area_px=min_area))


def _tune(scenes, grid, mask_fn):
    scores = np.zeros(len(grid))
    for sc in scenes:
        for i, thr in enumerate(grid):
            scores[i] += pixel_metrics(sc["truth_mask"], mask_fn(sc, thr))["f1"]
    return float(grid[int(np.argmax(scores))])


def tune_displacement(dem=None, seeds=TRAIN_SEEDS[:12], scenes=None):
    scenes = list(synthetic_scenes(dem, seeds)) if scenes is None else scenes
    return _tune(scenes, np.arange(5, 65, 5), displacement_mask)


def tune_threshold(dem=None, seeds=TRAIN_SEEDS[:12], scenes=None):
    scenes = list(synthetic_scenes(dem, seeds)) if scenes is None else scenes
    return _tune(scenes, np.round(np.arange(0.10, 0.51, 0.025), 3), threshold_mask)


def run_benchmark(checkpoint, dem, tuned_thr, disp_thr, seeds=TEST_SEEDS, scenes=None):
    det = DisasterDetector(checkpoint_path=checkpoint)
    if not det.is_trained:
        raise RuntimeError("No compatible U-Net checkpoint — train first.")
    scenes = synthetic_scenes(dem, seeds) if scenes is None else scenes
    rows = []
    for i, sc in enumerate(scenes):
        truth = sc["truth_mask"]
        x, _ = scene_input(sc, _wl(sc), INC_ANGLE)
        preds = {
            "U-Net": _valid(sc, det.postprocess(det.ensemble_predict(x), min_area_px=MIN_AREA)),
            f"Threshold Δγ>{DEFAULT_THR:.2f}": threshold_mask(sc, DEFAULT_THR),
            f"Threshold Δγ>{tuned_thr:.2f} (tuned)": threshold_mask(sc, tuned_thr),
            f"Displacement |d|>{disp_thr:.0f} mm (tuned)": displacement_mask(sc, disp_thr),
        }
        for name, pred in preds.items():
            n_true, hit, fa = object_metrics(truth, pred)
            rows.append(dict(seed=sc.get("seed", seeds[i] if seeds else i), method=name,
                             pair=sc.get("pair"), n_true=n_true, hit=hit, false_alarms=fa,
                             **pixel_metrics(truth, pred)))
    return rows


def summarise(rows):
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == m]
        f1 = np.array([r["f1"] for r in rs])
        out[m] = dict(
            f1_mean=float(f1.mean()), f1_std=float(f1.std()),
            precision=float(np.mean([r["precision"] for r in rs])),
            recall=float(np.mean([r["recall"] for r in rs])),
            landslides_found=int(sum(r["hit"] for r in rs)),
            landslides_total=int(sum(r["n_true"] for r in rs)),
            false_alarms=int(sum(r["false_alarms"] for r in rs)),
        )
    return out


def full_run(checkpoint, dem, out_json, epochs=25, log=print, train=True, real_dir=None):
    """Train + benchmark. With `real_dir` holding cached NISAR pairs, training also uses
    hybrid scenes (real background + injected landslides) and a second benchmark runs on
    hybrid scenes built from held-out dates."""
    t0 = time.time()
    tuned = tune_threshold(dem)
    disp_thr = tune_displacement(dem)
    log(f"tuned on synthetic train scenes: Δγ > {tuned}, |d| > {disp_thr} mm")

    train_pairs = test_pairs = None
    if real_dir:
        from .hybrid import hybrid_scenes, split_pairs
        from .nisar_asf import load_cached
        items = load_cached(real_dir)
        try:
            train_pairs, test_pairs = split_pairs(items)
        except RuntimeError as e:
            log(f"real data skipped: {e}")
    if train:
        scenes = list(synthetic_scenes(dem, TRAIN_SEEDS))
        meta = dict(train_seeds=TRAIN_SEEDS, epochs=epochs, labels="synthetic ground truth")
        if train_pairs:
            scenes += list(hybrid_scenes(train_pairs, dem, HYBRID_TRAIN_SEEDS))
            meta.update(labels="synthetic + real NISAR background with injected landslides",
                        real_train_pairs=[f"{c['ref_date']} → {c['sec_date']}"
                                          for _, c in train_pairs],
                        n_scenes=len(scenes))
            log(f"training on {len(TRAIN_SEEDS)} synthetic + "
                f"{len(train_pairs) * len(HYBRID_TRAIN_SEEDS)} real-background scenes")
        trainer = train_unet(scenes=scenes, epochs=epochs,
                             progress=lambda e, n, l: log(f"epoch {e}/{n} loss {l:.4f}"))
        trainer.save_checkpoint(checkpoint, meta=meta)
    rows = run_benchmark(checkpoint, dem, tuned, disp_thr)
    result = dict(summary=summarise(rows), rows=rows, tuned_threshold=tuned,
                  displacement_threshold=disp_thr,
                  default_threshold=DEFAULT_THR, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS,
                  epochs=epochs)
    if test_pairs:
        tr = list(hybrid_scenes(train_pairs, dem, HYBRID_TRAIN_SEEDS[:2]))
        r_thr, r_disp = tune_threshold(scenes=tr), tune_displacement(scenes=tr)
        log(f"tuned on real-background train scenes: Δγ > {r_thr}, |d| > {r_disp} mm")
        te = list(hybrid_scenes(test_pairs, dem, HYBRID_TEST_SEEDS))
        r_rows = run_benchmark(checkpoint, dem, r_thr, r_disp, seeds=None, scenes=te)
        result["real"] = dict(
            summary=summarise(r_rows), rows=r_rows, tuned_threshold=r_thr,
            displacement_threshold=r_disp,
            train_pairs=[f"{c['ref_date']} → {c['sec_date']}" for _, c in train_pairs],
            test_pairs=[f"{c['ref_date']} → {c['sec_date']}" for _, c in test_pairs],
            n_test_scenes=len(te))
    result["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    return result
