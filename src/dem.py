"""Real terrain for the study area.

We use the public AWS Terrain Tiles (Terrarium encoding, built from SRTM / ALOS World 3D),
because both the synthetic scenes and the 3D views should sit on the actual Tien Shan relief
around Almaty rather than on a made-up sine-wave surface. The mosaic is cached to
data/dem/ so the app works offline after the first download.
"""

from __future__ import annotations

import io
import math
import os

import numpy as np
from scipy.ndimage import map_coordinates

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def fetch_dem(bounds, shape=(512, 512), zoom: int = 11, cache_path: str | None = None):
    """Elevation (m) on a regular lat/lon grid. bounds = (lat_min, lat_max, lon_min, lon_max).

    Row 0 is north, matching the rest of the pipeline.
    """
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if tuple(z["bounds"]) == tuple(bounds) and z["dem"].shape == tuple(shape):
            return z["dem"]

    import requests
    from PIL import Image

    lat_min, lat_max, lon_min, lon_max = bounds
    x0, y0 = _lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = _lonlat_to_tile(lon_max, lat_min, zoom)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)

    mosaic = np.zeros((len(ty) * 256, len(tx) * 256), dtype=np.float32)
    for j, y in enumerate(ty):
        for i, x in enumerate(tx):
            r = requests.get(TILE_URL.format(z=zoom, x=x, y=y), timeout=30)
            r.raise_for_status()
            rgb = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32)
            elev = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = elev

    # Sample the Web-Mercator mosaic at the target lat/lon grid
    h, w = shape
    lats = np.linspace(lat_max, lat_min, h)
    lons = np.linspace(lon_min, lon_max, w)
    n = 2 ** zoom
    px = ((lons + 180.0) / 360.0 * n - tx[0]) * 256
    lat_r = np.radians(lats)
    py = ((1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / np.pi) / 2.0 * n - ty[0]) * 256
    PY, PX = np.meshgrid(py, px, indexing="ij")
    dem = map_coordinates(mosaic, [PY, PX], order=1, mode="nearest").astype(np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, dem=dem, bounds=np.array(bounds))
    return dem


def slope_deg(dem, bounds):
    """Terrain slope in degrees for a lat/lon grid DEM."""
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = dem.shape
    dy = (lat_max - lat_min) * 111_320 / (h - 1)
    dx = (lon_max - lon_min) * 111_320 * np.cos(np.radians((lat_min + lat_max) / 2)) / (w - 1)
    gy, gx = np.gradient(dem.astype(np.float64), dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
