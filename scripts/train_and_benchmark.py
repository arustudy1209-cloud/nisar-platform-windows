"""Train the U-Net and benchmark it against coherence / displacement thresholds.

    python scripts/train_and_benchmark.py [--epochs 25] [--synthetic-only]

If real NISAR pairs are cached in data/real/ (scripts/fetch_almaty.py), training adds
real-background scenes with injected landslides and a second benchmark runs on held-out
dates. See src/hybrid.py.

Writes data/sample/unet_checkpoint.pt and data/benchmark.json (shown in the app).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.benchmark import full_run  # noqa: E402
from src.dem import fetch_dem  # noqa: E402

AOI = (43.00, 43.40, 76.70, 77.20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--skip-train", action="store_true", help="benchmark the existing checkpoint")
    ap.add_argument("--synthetic-only", action="store_true", help="ignore data/real")
    a = ap.parse_args()
    dem = fetch_dem(AOI, cache_path=os.path.join(ROOT, "data", "dem", "almaty_dem.npz"))
    res = full_run(os.path.join(ROOT, "data", "sample", "unet_checkpoint.pt"), dem,
                   os.path.join(ROOT, "data", "benchmark.json"), epochs=a.epochs,
                   train=not a.skip_train,
                   real_dir=None if a.synthetic_only else os.path.join(ROOT, "data", "real"),
                   log=lambda m: print(m, flush=True))
    for k, v in res["summary"].items():
        print(f"{k:34s} F1={v['f1_mean']:.3f}±{v['f1_std']:.3f}  P={v['precision']:.3f} "
              f"R={v['recall']:.3f}  found {v['landslides_found']}/{v['landslides_total']}  "
              f"false alarms {v['false_alarms']}")
    if "real" in res:
        print("\nReal NISAR background, held-out dates:", ", ".join(res["real"]["test_pairs"]))
        for k, v in res["real"]["summary"].items():
            print(f"{k:34s} F1={v['f1_mean']:.3f}±{v['f1_std']:.3f}  P={v['precision']:.3f} "
                  f"R={v['recall']:.3f}  found {v['landslides_found']}/{v['landslides_total']}  "
                  f"false alarms {v['false_alarms']}")
    print("seconds", res["seconds"])
