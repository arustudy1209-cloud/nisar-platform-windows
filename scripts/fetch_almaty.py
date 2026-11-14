"""Fetch real NISAR GUNW data over Almaty — only the study-area subset of each file.

    # one-time: free account at https://urs.earthdata.nasa.gov, then EITHER
    export EARTHDATA_TOKEN=...                       # Profile -> Generate Token
    # OR
    export EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=...
    # OR a ~/.netrc line:  machine urs.earthdata.nasa.gov login USER password PASS

    python scripts/fetch_almaty.py              # time series on track 63 / frame 67 (100% AOI coverage)
    python scripts/fetch_almaty.py --browse     # only the public browse images (no login)

Saves data/real/NISAR_*.npz (~1-3 MB each) that the app picks up automatically.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.nisar_asf import (browse_crop, consecutive_series, fetch_subset,  # noqa: E402
                           have_credentials, search_gunw)

AOI = (43.00, 43.40, 76.70, 77.20)
REAL = os.path.join(ROOT, "data", "real")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=63)
    ap.add_argument("--frame", type=int, default=67)
    ap.add_argument("--browse", action="store_true", help="only public browse images")
    ap.add_argument("--max", type=int, default=0, help="limit number of pairs (0 = all)")
    a = ap.parse_args()

    gs = search_gunw(AOI, cache_path=os.path.join(REAL, "asf_search_cache.json"))
    chain = consecutive_series(gs, a.track, a.frame)
    print(f"{len(gs)} GUNW granules over Almaty; {len(chain)} consecutive pairs on "
          f"track {a.track} frame {a.frame}")
    for g in chain:
        browse_crop(g, AOI, os.path.join(REAL, "browse"))
    print("browse images cached")
    if a.browse:
        sys.exit(0)
    if not have_credentials():
        sys.exit("No Earthdata credentials — see the docstring at the top of this file.")
    todo = chain[-a.max:] if a.max else chain
    for g in todo:
        print(f"{g['ref_date']} -> {g['sec_date']}")
        fetch_subset(g, AOI, REAL)
    print("done:", REAL)
