"""Real NISAR GUNW data from the Alaska Satellite Facility (ASF DAAC).

* Search (ASF Search API) and browse images are public — no login.
* The GUNW HDF5 files (~2.5 GB each) need a free NASA Earthdata login. We never download
  the whole file: HDF5 is chunked, so we open it over HTTP with Range requests and read
  only the chunks that cover the study area (a few MB per granule).

Credentials, in order of preference:
  EARTHDATA_TOKEN env var (bearer token from urs.earthdata.nasa.gov -> Generate Token),
  EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars, or a ~/.netrc entry for
  urs.earthdata.nasa.gov.
"""

from __future__ import annotations

import io
import json
import os
import re
from collections import OrderedDict
from datetime import datetime

import numpy as np
import requests
from scipy.ndimage import map_coordinates

ASF_SEARCH = "https://api.daac.asf.alaska.edu/services/search/param"
URS_HOST = "urs.earthdata.nasa.gov"
NAME_RE = re.compile(
    r"NISAR_L2_\w\w_GUNW_(?P<cycle>\d{3})_(?P<track>\d{3})_(?P<dir>[AD])_(?P<frame>\d{3})_"
    r"(?P<cycle2>\d{3})_\d{4}_\w\w_(?P<r0>\d{8}T\d{6})_(?P<r1>\d{8}T\d{6})_"
    r"(?P<s0>\d{8}T\d{6})_(?P<s1>\d{8}T\d{6})_")


def aoi_wkt(bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    return (f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},"
            f"{lon_min} {lat_max},{lon_min} {lat_min}))")


def parse_name(name: str) -> dict:
    m = NAME_RE.search(name)
    if not m:
        return {}
    g = m.groupdict()
    ts = lambda s: datetime.strptime(s, "%Y%m%dT%H%M%S")
    return dict(track=int(g["track"]), frame=int(g["frame"]),
                direction="ASCENDING" if g["dir"] == "A" else "DESCENDING",
                ref_date=ts(g["r0"]).date(), sec_date=ts(g["s0"]).date())


# ---------------------------------------------------------------------------
# Search + browse (public)
# ---------------------------------------------------------------------------
def _size_mb(p):
    b = p.get("bytes")
    if isinstance(b, dict):
        b = next((v.get("bytes") for k, v in b.items() if k.endswith(".h5")), None)
    return round(b / 1e6) if isinstance(b, (int, float)) else None


def search_gunw(bounds, max_results: int = 250, cache_path: str | None = None) -> list[dict]:
    """GUNW granules intersecting the AOI, newest first. Falls back to cache offline."""
    try:
        r = requests.get(ASF_SEARCH, params=dict(dataset="NISAR", processingLevel="GUNW",
                                                 intersectsWith=aoi_wkt(bounds),
                                                 output="geojson", maxResults=max_results),
                         timeout=30)
        r.raise_for_status()
        feats = r.json()["features"]
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(feats, f)
    except Exception:
        if not (cache_path and os.path.exists(cache_path)):
            raise
        with open(cache_path) as f:
            feats = json.load(f)
    out = []
    for f in feats:
        p = f["properties"]
        info = parse_name(p["sceneName"])
        browse = [b for b in (p.get("browse") or []) if b.endswith("_LATLON.png")]
        out.append(dict(name=p["sceneName"], url=p["url"], browse=browse[0] if browse else None,
                        footprint=f["geometry"]["coordinates"][0],
                        size_mb=_size_mb(p),
                        **info))
    out.sort(key=lambda g: (g.get("sec_date") or datetime.min.date()), reverse=True)
    return out


def consecutive_series(granules, track: int, frame: int) -> list[dict]:
    """Chain of pairs on one track/frame where each secondary date = next reference date."""
    gs = [g for g in granules if g.get("track") == track and g.get("frame") == frame]
    by_ref = {}
    for g in sorted(gs, key=lambda g: g["name"]):          # keep the latest version per pair
        by_ref[g["ref_date"]] = g
    if not by_ref:
        return []
    chains = []
    for start in sorted(by_ref):
        chain, cur = [], start
        while cur in by_ref:
            chain.append(by_ref[cur])
            cur = by_ref[cur]["sec_date"]
        chains.append(chain)
    return max(chains, key=len)


def browse_crop(granule: dict, bounds, cache_dir: str):
    """Crop of the public LATLON browse PNG over the AOI (RGBA uint8) or None.

    The LATLON browse is drawn on a regular lat/lon grid spanning the footprint's bounding box.
    """
    if not granule.get("browse"):
        return None
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, granule["name"] + "_AOI.png")
    if os.path.exists(path):                       # only the AOI crop is kept on disk
        return np.asarray(Image.open(path).convert("RGBA"))
    r = requests.get(granule["browse"], timeout=60)
    r.raise_for_status()
    im = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    fp = np.asarray(granule["footprint"])
    lon0, lon1 = fp[:, 0].min(), fp[:, 0].max()
    lat0, lat1 = fp[:, 1].min(), fp[:, 1].max()
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = im.shape[:2]
    c0 = int(np.clip((lon_min - lon0) / (lon1 - lon0) * w, 0, w))
    c1 = int(np.clip((lon_max - lon0) / (lon1 - lon0) * w, 0, w))
    r0 = int(np.clip((lat1 - lat_max) / (lat1 - lat0) * h, 0, h))
    r1 = int(np.clip((lat1 - lat_min) / (lat1 - lat0) * h, 0, h))
    if c1 - c0 < 4 or r1 - r0 < 4:
        return None
    crop = im[r0:r1, c0:c1]
    Image.fromarray(crop).save(path)
    return crop


# ---------------------------------------------------------------------------
# Earthdata auth + HTTP range file
# ---------------------------------------------------------------------------
class _EDLSession(requests.Session):
    """Keeps basic auth only when talking to URS (requests drops it on redirects)."""

    def __init__(self, username=None, password=None):
        super().__init__()
        self._edl = (username, password) if username and password else None

    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        host = requests.utils.urlparse(prepared_request.url).hostname
        if self._edl and host == URS_HOST:
            prepared_request.prepare_auth(self._edl)


def have_credentials() -> bool:
    if os.environ.get("EARTHDATA_TOKEN"):
        return True
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return True
    try:
        import netrc
        return netrc.netrc().authenticators(URS_HOST) is not None
    except Exception:
        return False


def resolve_signed_url(url: str, token=None, username=None, password=None) -> str:
    """Follow ASF's Earthdata login dance and return the short-lived signed S3 URL."""
    token = token or os.environ.get("EARTHDATA_TOKEN")
    username = username or os.environ.get("EARTHDATA_USERNAME")
    password = password or os.environ.get("EARTHDATA_PASSWORD")
    if token:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                         allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 303, 307) and "earthdata" not in r.headers["location"]:
            return r.headers["location"]
        raise PermissionError(f"Earthdata token rejected (HTTP {r.status_code}).")
    s = _EDLSession(username, password)
    if not username:                                        # fall back to ~/.netrc
        import netrc
        auth = netrc.netrc().authenticators(URS_HOST) if os.path.exists(
            os.path.expanduser("~/.netrc")) else None
        if auth:
            s._edl = (auth[0], auth[2])
    if s._edl is None:
        raise PermissionError("No Earthdata credentials found.")
    r = s.get(url, stream=True, timeout=60)
    r.close()
    if r.status_code != 200:
        raise PermissionError(f"Earthdata login failed (HTTP {r.status_code}). Check the "
                              "password and that the ASF application is authorised in your "
                              "Earthdata profile.")
    return r.url


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP Range requests with a small block cache.
    h5py accepts any such file-like object."""

    def __init__(self, url: str, block: int = 2 * 1024 * 1024, max_blocks: int = 96,
                 refresh=None):
        self.url, self.block, self.max_blocks, self.refresh = url, block, max_blocks, refresh
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off,
                    io.SEEK_END: self.size + off}[whence]
        return self.pos

    def _get_block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        start, end = i * self.block, min(self.size, (i + 1) * self.block) - 1
        r = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        if r.status_code == 403 and self.refresh:          # signed URL expired
            self.url = self.refresh()
            r = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        r.raise_for_status()
        data = r.content
        self.bytes_fetched += len(data)
        self.cache[i] = data
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block)
            chunk = self._get_block(i)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# ---------------------------------------------------------------------------
# GUNW reader
# ---------------------------------------------------------------------------
GRID_CANDIDATES = [
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram",
    "/science/LSAR/GUNW/grids/frequencyA/interferogram/unwrapped",
]


def _find_grid(f):
    """Locate the unwrapped-interferogram group, polarisation subgroup and coordinates."""
    import h5py

    grid = next((g for g in GRID_CANDIDATES if g in f), None)
    if grid is None:                                        # schema changed: search for it
        hits = []
        f.visititems(lambda p, o: hits.append(p) if isinstance(o, h5py.Dataset)
                     and p.endswith("unwrappedPhase") else None)
        if not hits:
            raise RuntimeError("No unwrappedPhase dataset in this file.")
        pol_path = "/" + hits[0].rsplit("/", 1)[0]
        grid = pol_path.rsplit("/", 1)[0]
    g = f[grid]
    pols = [k for k in g if isinstance(g[k], h5py.Group) and "unwrappedPhase" in g[k]]
    if not pols:
        raise RuntimeError(f"No polarisation group with unwrappedPhase under {grid}.")
    pol = "HH" if "HH" in pols else pols[0]
    holder = g if "xCoordinates" in g else g[pol]
    return g, g[pol], pol, holder


def _epsg(f, holder, lon_c):
    for key in ("projection",):
        if key in holder:
            ds = holder[key]
            for a in ("epsg_code", "epsg"):
                if a in ds.attrs:
                    return int(np.asarray(ds.attrs[a]).ravel()[0])
            try:
                return int(ds[()])
            except Exception:
                pass
    zone = int((lon_c + 180) // 6) + 1                      # Almaty -> UTM 43N
    return 32600 + zone


def read_gunw_subset(fileobj, bounds, shape=(512, 512)) -> dict:
    """Read unwrapped phase, coherence and connected components over `bounds` and resample
    to a regular lat/lon grid (row 0 = north), the grid every other part of the app uses."""
    import h5py
    from pyproj import Transformer

    lat_min, lat_max, lon_min, lon_max = bounds
    with h5py.File(fileobj, "r") as f:
        grid, polg, pol, holder = _find_grid(f)
        x = holder["xCoordinates"][()]
        y = holder["yCoordinates"][()]
        epsg = _epsg(f, holder, (lon_min + lon_max) / 2)

        h, w = shape
        lats = np.linspace(lat_max, lat_min, h)
        lons = np.linspace(lon_min, lon_max, w)
        LON, LAT = np.meshgrid(lons, lats)
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        X, Y = tr.transform(LON, LAT)

        # Fractional native-grid indices of every target pixel, then a window around them
        colf = (X - x[0]) / (x[1] - x[0])
        rowf = (Y - y[0]) / (y[1] - y[0])
        c0, c1 = int(max(0, np.floor(colf.min()) - 4)), int(min(len(x), np.ceil(colf.max()) + 5))
        r0, r1 = int(max(0, np.floor(rowf.min()) - 4)), int(min(len(y), np.ceil(rowf.max()) + 5))
        if c1 - c0 < 2 or r1 - r0 < 2:
            raise RuntimeError("The study area is outside this granule's grid.")

        def win(name):
            return polg[name][r0:r1, c0:c1].astype(np.float32) if name in polg else None

        phase = win("unwrappedPhase")
        coh = win("coherenceMagnitude")
        cc = win("connectedComponents")
        iono = win("ionospherePhaseScreen")
        freq = None
        for p in (grid.name.rsplit("/", 1)[0] + "/centerFrequency",
                  grid.name.rsplit("/", 2)[0] + "/centerFrequency"):
            if p in f:
                freq = float(np.asarray(f[p][()]).ravel()[0])
                break

    col, row = colf - c0, rowf - r0
    samp = lambda a, o=1: map_coordinates(a, [row, col], order=o, mode="constant", cval=np.nan)

    for a in (phase, coh, iono):
        if a is not None:
            a[~np.isfinite(a)] = np.nan
    if iono is not None and np.isfinite(iono).any():       # L-band: ionosphere matters
        phase = phase - np.nan_to_num(iono)
    valid = np.isfinite(phase) & (phase != 0)
    if cc is not None:
        valid &= cc > 0
    phase = np.where(valid, phase, np.nan)

    out_phase = samp(np.nan_to_num(phase))
    out_valid = samp(valid.astype(np.float32)) > 0.5
    out_coh = np.clip(np.nan_to_num(samp(np.nan_to_num(coh)) if coh is not None
                                    else out_valid.astype(np.float32)), 0, 1)
    # Reference the phase to the median of coherent valid pixels
    ref_mask = out_valid & (out_coh > 0.6)
    if ref_mask.sum() > 100:
        out_phase = out_phase - np.median(out_phase[ref_mask])
    out_phase[~out_valid] = 0.0
    return dict(unwrapped_phase=out_phase.astype(np.float32),
                coherence=out_coh.astype(np.float32), valid=out_valid,
                wavelength=(299_792_458.0 / freq) if freq else 0.2384,
                epsg=epsg, polarization=pol, native_window=(r0, r1, c0, c1))


def fetch_subset(granule: dict, bounds, out_dir: str, shape=(512, 512), log=print) -> str:
    """Download only the AOI subset of one granule into out_dir/<name>.npz."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, granule["name"] + ".npz")
    if os.path.exists(out):
        return out
    signed = resolve_signed_url(granule["url"])
    fobj = HTTPRangeFile(signed, refresh=lambda: resolve_signed_url(granule["url"]))
    sub = read_gunw_subset(fobj, bounds, shape)
    log(f"  {granule['name'][:60]}…  read {fobj.bytes_fetched / 1e6:.1f} MB of "
        f"{fobj.size / 1e9:.2f} GB")
    np.savez_compressed(out, **sub, bounds=np.array(bounds), name=granule["name"],
                        ref_date=str(granule["ref_date"]), sec_date=str(granule["sec_date"]),
                        track=granule["track"], frame=granule["frame"])
    return out


def load_cached(out_dir: str) -> list[dict]:
    if not os.path.isdir(out_dir):
        return []
    items = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".npz") and fn.startswith("NISAR_"):
            z = np.load(os.path.join(out_dir, fn), allow_pickle=False)
            items.append({k: z[k] for k in z.files})
    for it in items:
        it["ref_date"] = datetime.strptime(str(it["ref_date"]), "%Y-%m-%d").date()
        it["sec_date"] = datetime.strptime(str(it["sec_date"]), "%Y-%m-%d").date()
        it["name"] = str(it["name"])
    items.sort(key=lambda d: d["sec_date"])
    return items
