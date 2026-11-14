# NISAR Disaster Detection Platform

Landslide and ground-subsidence monitoring for **Almaty and the northern Tien Shan foothills**
from NISAR L-band interferograms (GUNW). Built for NASA Space Apps Challenge 2026 —
*Dancing with the SARs*.

What it does:

- **Real NISAR data.** Searches ASF for GUNW products over Almaty (46 granules, 6
  track/frames). It then reads *only the Almaty window* out of each 2.5 GB HDF5 file over HTTP
  range requests, applies the ionospheric phase screen, and resamples to a lat/lon grid.
- **Detection.** A U-Net with 4 input channels (previous coherence, current coherence,
  amplitude, high-passed displacement), benchmarked against a plain coherence threshold on the
  same held-out scenes.
- **Forecast.** Inverse-velocity method (Fukuzono) on a stack of consecutive 12-day pairs. It
  predicts a failure-date window for accelerating slopes and maps that window to warning levels.
- **3D.** An interactive globe (globe.gl, Esri imagery) with computed risk columns and the real
  NISAR frame outlines, and a 3D model of the real Almaty relief draped with any computed
  layer. Hover shows values.

## Benchmark (20 held-out simulated scenes, 93 landslides)

| Method | F1 | Landslides found | False alarms |
|---|---|---|---|
| U-Net (ours) | 0.945 ± 0.012 | 93 / 93 | 0 |
| Displacement threshold \|d\| > 10 mm (tuned) | 0.816 | 93 / 93 | 23 |
| Coherence threshold Δγ > 0.20 (tuned) | 0.253 | 55 / 93 | 87 |
| Coherence threshold Δγ > 0.30 | 0.235 | 50 / 93 | 65 |

Thresholds were tuned on the training scenes, and all methods share the same post-processing.
These scores come from **simulated** scenes on the real Almaty DEM. They include snow and field
decorrelation, slow coherent landslides and topography-correlated atmosphere. Real-world
scores will be lower until we validate against a landslide inventory. Reproduce with
`python scripts/train_and_benchmark.py`.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Real NISAR data

Search and browse images are public. Downloading GUNW data requires a free
[NASA Earthdata](https://urs.earthdata.nasa.gov) account:

```
export EARTHDATA_USERNAME=...  EARTHDATA_PASSWORD=...   # or EARTHDATA_TOKEN=..., or ~/.netrc
python scripts/fetch_almaty.py      # 9 consecutive pairs, track 63 / frame 67, a few MB each
```

The app picks up `data/real/NISAR_*.npz` automatically ("Real NISAR · Almaty" data source).
We use descending track 63 / frame 67 because it covers 100% of the study area. Track 127 /
frame 23 covers only ~45%, because Almaty sits at its swath edge.

## Layout

```
app.py                     Streamlit UI
src/nisar_asf.py           ASF search, Earthdata auth, HTTP range HDF5 reader, GUNW subset
src/physics.py             InSAR conversions, scene + time-series simulator
src/features.py            model inputs (shared by simulated and real data)
src/model.py, inference.py U-Net, training, TTA, post-processing, event polygons
src/benchmark.py           U-Net vs threshold protocol
src/forecast.py            cumulative displacement, inverse velocity, warning levels
src/globe.py, visualization.py   3D globe, 3D terrain, maps
src/dem.py                 real terrain (AWS Terrain Tiles), cached in data/dem/
scripts/                   fetch_almaty.py, train_and_benchmark.py
```

Team: Aruzhan Lastayeva, Aisha Karimova, Taniris Sarseken.
