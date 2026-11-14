"""NISAR Disaster Detection Platform — Streamlit app.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

from src.benchmark import object_metrics, pixel_metrics
from src.coherence import CoherenceAnalyzer
from src.dem import fetch_dem
from src.features import build_input
from src.forecast import (acceleration_map, cumulative_displacement, inverse_velocity,
                          moving_zones, warning_level)
from src.globe import RISK_STOPS, build_globe_html, colorize, png_data_uri
from src.inference import DisasterDetector
from src.nisar_asf import (browse_crop, consecutive_series, have_credentials, load_cached,
                           read_gunw_subset, search_gunw)
from src.physics import NISAR_L_WAVELENGTH, SARPhysics
from src.visualization import DisasterMap, terrain_3d

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(APP_DIR, "data")
CHECKPOINT = os.path.join(DATA, "sample", "unet_checkpoint.pt")
BENCH_JSON = os.path.join(DATA, "benchmark.json")
REAL_DIR = os.path.join(DATA, "real")
DEM_CACHE = os.path.join(DATA, "dem", "almaty_dem.npz")
ASF_CACHE = os.path.join(REAL_DIR, "asf_search_cache.json")

# Study area: Almaty and the northern Tien Shan foothills (Medeu district)
AOI = (43.00, 43.40, 76.70, 77.20)  # lat_min, lat_max, lon_min, lon_max
REAL_TRACK, REAL_FRAME = 63, 67     # descending; the only frame with 100% AOI coverage + long chain
SITES = [  # approximate locations of documented landslide-prone slopes
    dict(name="Kamenskoe plateau", lat=43.185, lng=76.950,
         note="2024 landslide, family of four killed; slopes since reinforced"),
    dict(name="Kok-Tobe slope", lat=43.231, lng=76.977,
         note="emergency slope, reinforcement works"),
    dict(name="Sunkar / Esentai City", lat=43.196, lng=76.905,
         note="landslide forming above a cottage complex"),
]
TEAM = ["Aruzhan Lastayeva", "Aisha Karimova", "Taniris Sarseken"]

st.set_page_config(page_title="NISAR Disaster Detection Platform",
                   page_icon=":material/satellite_alt:", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, p, li, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stButton button, .stTabs button {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.2rem; max-width: 1500px; }
div[data-testid="stMetric"] {
  border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; padding: 10px 14px;
  background: rgba(255,255,255,0.02);
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
  font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] p { font-size: 0.8rem !important; color: #9aa4b1 !important; }
.kicker { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: #8a96a5; margin-bottom: 2px; }
.facts { color: #9aa4b1; font-size: 14px; margin-top: -6px; }
.facts b { color: #d7dde4; font-weight: 500; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 4px; font-weight: 600; color: #111; font-size: 14px; }
.note { color:#9aa4b1; font-size: 13px; }
table.plain { border-collapse: collapse; width: 100%; font-size: 14px; }
table.plain td, table.plain th { border-bottom: 1px solid rgba(255,255,255,0.08); padding: 6px 8px; text-align: left; vertical-align: top; }
table.plain th { color: #9aa4b1; font-weight: 500; }
/* Streamlit draws icons with the Material Symbols ligature font — keep it */
html body span[data-testid="stIconMaterial"], html body [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important; font-weight: 400 !important;
  letter-spacing: normal !important;
}
</style>
""", unsafe_allow_html=True)

PLOT_FONT = dict(family="IBM Plex Sans, system-ui, sans-serif", size=12, color="#cfd6df")
LVL_HEX = {"LOW": "#2ea06e", "MEDIUM": "#f0be28", "HIGH": "#e86e28", "CRITICAL": "#d62828"}
WARN_HEX = {"NORMAL": "#2ea06e", "WATCH": "#f0be28", "WARNING": "#e86e28", "ALERT": "#d62828"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_dem():
    try:
        return fetch_dem(AOI, cache_path=DEM_CACHE)
    except Exception:
        return None  # offline and no cache: the simulator falls back to procedural terrain


@st.cache_data(show_spinner=False)
def load_sim(seed: int):
    return SARPhysics.generate_synthetic_gunw(seed=seed, dem=get_dem())


@st.cache_data(show_spinner=False, ttl=3600)
def asf_granules():
    try:
        return search_gunw(AOI, cache_path=ASF_CACHE)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def real_items(_mtime: float):
    return load_cached(REAL_DIR)


def real_chain(items):
    """Longest run of cached pairs where each secondary date = next reference date."""
    if not items:
        return []
    chain = [items[0]]
    best = chain
    for it in items[1:]:
        chain = chain + [it] if it["ref_date"] == chain[-1]["sec_date"] else [it]
        if len(chain) > len(best):
            best = chain
    return best


def proxy_amplitude(shape, dem):
    """GUNW products carry no amplitude layer; use the same terrain-driven backscatter model
    the network was trained with."""
    rng = np.random.default_rng(0)
    _, slope, gx, _ = SARPhysics._terrain(shape, rng, dem)
    return SARPhysics._amplitude(slope, gx, rng, shape).astype(np.float32)


def real_scene(items, idx, dem):
    it = items[idx]
    coh_after = it["coherence"].astype(np.float32)
    valid = it["valid"].astype(bool)
    prev = next((p for p in items if p["sec_date"] == it["ref_date"]), None)
    if prev is not None:
        coh_before = prev["coherence"].astype(np.float32)
        valid &= prev["valid"].astype(bool)
    else:
        coh_before = np.maximum(coh_after, gaussian_filter(coh_after, 15) + 0.1).clip(0, 1)
    coh_before = np.where(valid, coh_before, 0)
    coh_after = np.where(valid, coh_after, 0)
    phase = np.where(valid, it["unwrapped_phase"], 0).astype(np.float32)
    dem_ = dem if dem is not None else np.zeros_like(phase)
    return {
        "unwrapped_phase": phase, "wrapped_phase": SARPhysics.wrap_phase(phase),
        "coherence_before": coh_before, "coherence_after": coh_after,
        "amplitude": proxy_amplitude(phase.shape, dem), "dem": dem_, "truth_mask": None,
        "valid": valid, "wavelength": float(it["wavelength"]),
        "label": f"NISAR GUNW {it['ref_date']} → {it['sec_date']} · track {int(it['track'])} "
                 f"frame {int(it['frame'])}",
    }


def load_uploaded(file) -> dict:
    """NISAR GUNW .h5 (read with the same subset reader as ASF data) or a .npz."""
    raw = file.getvalue()
    if file.name.lower().endswith(".npz"):
        z = np.load(io.BytesIO(raw))
        phase = z["unwrapped_phase"].astype(np.float32)
        coh_after = np.clip(z["coherence_after"] if "coherence_after" in z else z["coherence"], 0, 1)
        coh_before = z["coherence_before"] if "coherence_before" in z else None
        valid = np.ones(phase.shape, bool)
    else:
        sub = read_gunw_subset(io.BytesIO(raw), AOI)
        phase, coh_after, coh_before, valid = (sub["unwrapped_phase"], sub["coherence"], None,
                                               sub["valid"])
    if coh_before is None:
        coh_before = np.maximum(coh_after, gaussian_filter(coh_after, 15) + 0.1).clip(0, 1)
    dem = get_dem()
    if dem is None or dem.shape != phase.shape:
        dem = np.zeros_like(phase)
    return {"unwrapped_phase": phase, "wrapped_phase": SARPhysics.wrap_phase(phase),
            "coherence_before": coh_before.astype(np.float32),
            "coherence_after": coh_after.astype(np.float32),
            "amplitude": proxy_amplitude(phase.shape, dem if dem.any() else None),
            "dem": dem, "truth_mask": None, "valid": valid, "label": f"Uploaded: {file.name}"}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_detector(_mtime: float):
    return DisasterDetector(checkpoint_path=CHECKPOINT)


def ckpt_mtime():
    return os.path.getmtime(CHECKPOINT) if os.path.exists(CHECKPOINT) else 0.0


def run_analysis(data, wavelength, inc_angle, coh_thr, min_area):
    los = SARPhysics.phase_to_displacement(data["unwrapped_phase"], wavelength)
    vert = SARPhysics.compute_los_to_vertical(los, inc_angle)
    dcoh = CoherenceAnalyzer.compute_coherence_change(data["coherence_before"],
                                                      data["coherence_after"])
    valid = data.get("valid")
    valid = np.ones(vert.shape, bool) if valid is None else valid
    vert = np.where(valid, vert, 0.0)
    dcoh = np.where(valid, dcoh, 0.0)
    risk_map = CoherenceAnalyzer.compute_risk_score(SARPhysics.highpass(vert), dcoh)

    detector = get_detector(ckpt_mtime())
    x = build_input(data["coherence_before"], data["coherence_after"], data["amplitude"], vert)
    if detector.is_trained:
        prob = detector.ensemble_predict(x)
    else:  # before the first training: physics-based probability from the risk map
        prob = gaussian_filter(np.clip((risk_map - 30) / 40, 0, 1), 2)
    prob = np.where(valid, prob, 0)
    ml_mask = detector.postprocess(prob, min_area_px=min_area) & valid
    # The baseline gets the same post-processing, so the comparison is about the signal only
    coh_mask = detector.postprocess(dcoh > coh_thr, min_area_px=min_area) & valid
    events = detector.extract_event_polygons(ml_mask, AOI, vert, dcoh)

    truth = data.get("truth_mask")
    metrics = None
    if truth is not None:
        metrics = {}
        for name, m in (("U-Net", ml_mask), (f"Threshold Δγ>{coh_thr:.2f}", coh_mask)):
            n_true, hit, fa = object_metrics(truth, m)
            metrics[name] = dict(**pixel_metrics(truth, m), n_true=n_true, hit=hit,
                                 false_alarms=fa)
    return dict(los=los, vert=vert, dcoh=dcoh, coh_mask=coh_mask, risk_map=risk_map, prob=prob,
                ml_mask=ml_mask, events=events, metrics=metrics, valid=valid,
                model_used=detector.is_trained)


def site_check(R, data):
    """Model output around each known site (500 m radius)."""
    h, w = R["vert"].shape
    lat_min, lat_max, lon_min, lon_max = AOI
    yy, xx = np.mgrid[0:h, 0:w]
    px_m = (lat_max - lat_min) * 111_320 / h
    rows = []
    for s in SITES:
        r = (lat_max - s["lat"]) / (lat_max - lat_min) * (h - 1)
        c = (s["lng"] - lon_min) / (lon_max - lon_min) * (w - 1)
        m = (np.hypot(yy - r, xx - c) * px_m <= 500) & R["valid"]
        if not m.any():
            rows.append(dict(Site=s["name"], Note="no valid data"))
            continue
        rows.append({
            "Site": s["name"],
            "Max U-Net probability": round(float(R["prob"][m].max()), 2),
            "Detected": "yes" if R["ml_mask"][m].any() else "no",
            "Mean vertical disp. (mm)": round(float(SARPhysics.highpass(R["vert"])[m].mean()), 1),
            "Coherence after": round(float(data["coherence_after"][m].mean()), 2),
            "Δγ": round(float(R["dcoh"][m].mean()), 2),
            "Risk (0-100)": round(float(R["risk_map"][m].max()), 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def style(fig, height=360, title=None):
    fig.update_layout(template="plotly_dark", height=height, font=PLOT_FONT,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=34 if title else 8, b=0),
                      title=dict(text=title, font=dict(size=14)) if title else None)
    return fig


def image_fig(arr, title, cmap, zmin=None, zmax=None, cbar_title="", height=360):
    fig = px.imshow(arr, color_continuous_scale=cmap, zmin=zmin, zmax=zmax, aspect="equal")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(coloraxis_colorbar=dict(title=cbar_title, thickness=10))
    return style(fig, height, title)


def rgb_fig(rgb, title, height=380):
    fig = px.imshow(rgb)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return style(fig, height, title)


def error_rgb(amplitude, pred, truth=None):
    """Grey amplitude with TP green, FP red, FN amber (or just red if no truth)."""
    base = np.repeat((gaussian_filter(amplitude, 1) * 0.7)[..., None], 3, 2)
    out = base.copy()
    if truth is None:
        out[pred] = out[pred] * 0.3 + np.array([0.91, 0.43, 0.16]) * 0.7
    else:
        for m, col in ((pred & truth, (0.18, 0.63, 0.43)), (pred & ~truth, (0.84, 0.16, 0.16)),
                       (~pred & truth, (0.94, 0.75, 0.16))):
            out[m] = out[m] * 0.25 + np.array(col) * 0.75
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def frame_footprints(granules):
    from matplotlib.path import Path

    LA, LO = np.meshgrid(np.linspace(AOI[0], AOI[1], 30), np.linspace(AOI[2], AOI[3], 30))
    pts = np.c_[LO.ravel(), LA.ravel()]
    frames = {}
    for g in granules:
        k = (g.get("track"), g.get("frame"))
        if k in frames:
            frames[k]["n"] += 1
            continue
        fp = np.asarray(g["footprint"])
        frames[k] = dict(coords=fp.tolist(), track=k[0], frame=k[1],
                         direction=g.get("direction", ""), n=1, latest=str(g.get("sec_date")),
                         coverage=int(round(Path(fp).contains_points(pts).mean() * 100)))
    return list(frames.values())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
items_all = real_items(max([os.path.getmtime(os.path.join(REAL_DIR, f))
                            for f in os.listdir(REAL_DIR)] + [0]) if os.path.isdir(REAL_DIR) else 0)
SRC_SIM, SRC_REAL, SRC_UP = "Simulated scene · Almaty terrain", "Real NISAR · Almaty", "Upload GUNW file"

with st.sidebar:
    st.markdown("### NISAR Disaster Platform")
    st.caption("L-band InSAR · landslides & subsidence · Almaty")
    sources = [SRC_SIM] + ([SRC_REAL] if items_all else []) + [SRC_UP]
    source = st.radio("Data source", sources)
    if not items_all:
        st.caption("Real NISAR subsets not downloaded yet — see the **Real NISAR** tab.")

    seed, pair_idx, uploaded = 42, None, None
    if source == SRC_SIM:
        seed = int(st.number_input("Scene seed", 0, 9999, 42,
                                   help="Each seed is a different simulated 12-day pair"))
    elif source == SRC_REAL:
        labels = [f"{it['ref_date']} → {it['sec_date']}" for it in items_all]
        pair_idx = st.selectbox("Interferogram pair", range(len(items_all)),
                                index=len(items_all) - 1, format_func=lambda i: labels[i])
    else:
        uploaded = st.file_uploader("NISAR GUNW (.h5) or .npz", type=["h5", "npz"])

    st.markdown("**Sensor**")
    wavelength = st.number_input("Wavelength (m)", 0.01, 0.5, NISAR_L_WAVELENGTH, 0.001,
                                 format="%.4f", help="NISAR L-band 0.2384 m, S-band 0.094 m")
    inc_angle = st.slider("Incidence angle (°)", 30.0, 47.0, 37.0, 0.5,
                          help="NISAR swath spans ~33–47°; Almaty sits mid-swath on track 63")
    st.markdown("**Detection**")
    coh_thr = st.slider("Baseline: coherence-loss threshold Δγ", 0.1, 0.6, 0.30, 0.05)
    min_area = st.slider("Min event area (px, 1 px ≈ 80 m)", 10, 400, 50, 10)

    with st.expander("Model training"):
        st.caption("Trains the U-Net on 24 simulated scenes over the real Almaty DEM plus real "
                   "NISAR pairs (cached in data/real) with injected landslides, then re-runs both "
                   "benchmarks on held-out scenes and dates (~10 min on a laptop).")
        epochs = st.slider("Epochs", 5, 40, 25)
        train_btn = st.button("Retrain + benchmark", width="stretch")
    det = get_detector(ckpt_mtime())
    st.caption((f"Model: U-Net trained on {det.meta.get('n_scenes', len(det.meta.get('train_seeds', [])))} "
                f"scenes ({len(det.meta.get('real_train_pairs', []))} real NISAR pairs), "
                f"{det.meta.get('epochs', '?')} epochs") if det.is_trained
               else "No compatible model yet — physics baseline in use.")

if train_btn:
    from src.benchmark import full_run
    with st.sidebar:
        box = st.empty()
        full_run(CHECKPOINT, get_dem(), BENCH_JSON, epochs=epochs, real_dir=REAL_DIR,
                 log=lambda m: box.caption(m))
    st.cache_resource.clear()
    st.rerun()

# Load data
data = None
if source == SRC_UP:
    if uploaded is not None:
        try:
            data = load_uploaded(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    if data is None:
        st.info("Upload a NISAR GUNW .h5 file in the sidebar (only the Almaty window is read), "
                "or choose another data source.")
        st.stop()
elif source == SRC_REAL:
    data = real_scene(items_all, pair_idx, get_dem())
    wavelength = data["wavelength"]
else:
    data = dict(load_sim(seed), label=f"Simulated 12-day pair · seed {seed} · real Almaty DEM")

params = (source, seed, pair_idx, getattr(uploaded, "name", None), wavelength, inc_angle,
          coh_thr, min_area, ckpt_mtime())
if st.session_state.get("params") != params:
    with st.spinner("Running the InSAR pipeline…"):
        st.session_state.results = run_analysis(data, wavelength, inc_angle, coh_thr, min_area)
        st.session_state.params = params
R = st.session_state.results
feats = R["events"]["features"]
overall = max((f["properties"]["risk_score"] for f in feats), default=0.0)
level = CoherenceAnalyzer.risk_level(overall)
bench = json.load(open(BENCH_JSON)) if os.path.exists(BENCH_JSON) else None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>NASA Space Apps Challenge 2026 · Dancing with the SARs</div>",
            unsafe_allow_html=True)
st.markdown("## NISAR Disaster Detection Platform")
st.markdown(f"<div class='facts'>Study area <b>Almaty & Tien Shan foothills</b> · "
            f"NISAR L-band λ = <b>{wavelength * 100:.1f} cm</b> · 12-day repeat · "
            f"now showing: <b>{data['label']}</b></div>", unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["Overview", "Interferogram", "3D Terrain", "Detection: U-Net vs threshold",
                "Forecast", "Real NISAR data", "Method & Team"])

# --- Overview ---------------------------------------------------------------
with tabs[0]:
    px_km2 = DisasterDetector.pixel_area_km2(R["ml_mask"].shape, AOI)
    m = st.columns(5)
    m[0].metric("Detected events", len(feats))
    m[1].metric("Affected area", f"{R['ml_mask'].sum() * px_km2:.1f} km²")
    m[2].metric("Max subsidence", f"{np.percentile(R['vert'][R['valid']], 0.1):.0f} mm")
    m[3].metric("Overall risk", f"{overall:.0f} / 100", level, delta_color="off")
    if bench:
        s = bench["summary"]
        thr_key = next(k for k in s if k.startswith("Threshold") and "tuned" in k)
        m[4].metric("F1 · U-Net vs threshold", f"{s['U-Net']['f1_mean']:.2f} vs "
                                               f"{s[thr_key]['f1_mean']:.2f}")

    risk_rgb = colorize(R["risk_map"], 0, 100, RISK_STOPS)
    thumb = png_data_uri(risk_rgb[::3, ::3])
    granules = asf_granules()
    stats = dict(source=data["label"], n_events=len(feats), level=level,
                 overall=f"{overall:.0f}", max_sub=f"{np.percentile(R['vert'], 0.1):.0f}",
                 area=f"{R['ml_mask'].sum() * px_km2:.1f}")
    st.iframe(build_globe_html(R["risk_map"], AOI, R["events"], frame_footprints(granules),
                               SITES, thumb, stats), height=630)
    st.caption("3D globe: hex columns are our computed risk on a ~1.2 km H3 grid; blue outlines "
               "are the real NISAR GUNW frames that ASF holds over Almaty; white diamonds are "
               "documented landslide-prone slopes. Drag to rotate, scroll to zoom.")

# --- Interferogram ----------------------------------------------------------
with tabs[1]:
    cols = st.columns(3)
    layers = [("Wrapped phase", data["wrapped_phase"], "Twilight", "rad"),
              ("Coherence, previous pair", data["coherence_before"], "Inferno", "γ"),
              ("Coherence, this pair", data["coherence_after"], "Inferno", "γ")]
    for col, (t, arr, cmap, unit) in zip(cols, layers):
        with col:
            st.plotly_chart(image_fig(arr, t, cmap, cbar_title=unit), width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        hp = SARPhysics.highpass(R["vert"])
        lim = float(np.percentile(np.abs(hp), 99.5))
        st.plotly_chart(image_fig(hp, "Vertical displacement, high-passed", "RdBu", -lim, lim,
                                  "mm"), width="stretch")
    with c2:
        st.plotly_chart(image_fig(R["dcoh"], "Coherence loss Δγ", "Magma", 0, 0.6, "Δγ"),
                        width="stretch")
    with c3:
        st.plotly_chart(image_fig(data["dem"], "Terrain (SRTM / AW3D30)", "Earth", cbar_title="m"),
                        width="stretch")
    st.caption("Displacement = unwrapped phase × λ / 4π, projected from line of sight to vertical "
               "and high-passed (1.6 km Gaussian) to remove orbit ramps and large-scale "
               "atmosphere. Landslides here are < 2 km across, so they survive the filter.")

# --- 3D terrain -------------------------------------------------------------
with tabs[2]:
    c1, c2 = st.columns([3, 1])
    with c2:
        layer = st.radio("Layer draped on the terrain",
                         ["Risk score", "Vertical displacement", "U-Net probability",
                          "Coherence loss"], key="layer3d")
        exag = st.slider("Vertical exaggeration", 1.0, 3.0, 1.6, 0.1)
        st.caption("Real relief of the Almaty foothills (AWS Terrain Tiles). "
                   "Hover anywhere on the surface for the calculated value at that point.")
    spec = {
        "Risk score": (R["risk_map"], "Risk", "0-100",
                       [[p, f"rgb{c}"] for p, c in RISK_STOPS], 0, 100),
        "Vertical displacement": (SARPhysics.highpass(R["vert"]), "Vertical disp.", "mm",
                                  "RdBu", -40, 40),
        "U-Net probability": (R["prob"], "P(landslide)", "", "Viridis", 0, 1),
        "Coherence loss": (R["dcoh"], "Δγ", "", "Magma", 0, 0.6),
    }[layer]
    with c1:
        st.plotly_chart(terrain_3d(data["dem"], spec[0], AOI, spec[1], spec[2], spec[3], spec[4],
                                   spec[5], events=R["events"], sites=SITES, exaggeration=exag),
                        width="stretch")
    with st.expander("Flat map with event outlines (pydeck)"):
        st.pydeck_chart(DisasterMap().create_3d_map(R["vert"], R["ml_mask"], data["dem"], AOI,
                                                    events=R["events"]), height=560)

# --- Detection: U-Net vs threshold -------------------------------------------
with tabs[3]:
    truth = data.get("truth_mask")
    if not R["model_used"]:
        st.warning("No compatible U-Net checkpoint — the U-Net column shows the physics baseline. "
                   "Use **Model training** in the sidebar.")
    st.markdown("#### This scene, side by side")
    ncol = 3 if truth is not None else 2
    cols = st.columns(ncol)
    if truth is not None:
        with cols[0]:
            st.plotly_chart(rgb_fig(error_rgb(data["amplitude"], truth.astype(bool)),
                                    "Ground truth (simulated)"), width="stretch")
            st.caption("Real slope movement only. Snow and field decorrelation are not landslides.")
    for col, (name, mask) in zip(cols[-2:], (("U-Net", R["ml_mask"]),
                                             (f"Threshold Δγ>{coh_thr:.2f}", R["coh_mask"]))):
        with col:
            st.plotly_chart(rgb_fig(error_rgb(data["amplitude"], mask, truth), name),
                            width="stretch")
            if R["metrics"]:
                mt = R["metrics"][name]
                st.markdown(f"F1 **{mt['f1']:.2f}** · precision {mt['precision']:.2f} · "
                            f"recall {mt['recall']:.2f}  \nlandslides found **{mt['hit']}/"
                            f"{mt['n_true']}** · false alarms **{mt['false_alarms']}**")
    st.caption("Green = correct, red = false alarm, amber = missed." if truth is not None else
               "Real data has no ground truth; see the known-site check in **Real NISAR data**.")

    st.divider()
    st.markdown("#### Benchmark on 20 held-out scenes")
    if bench:
        s = bench["summary"]
        df = pd.DataFrame([{"Method": k, "F1": f"{v['f1_mean']:.3f} ± {v['f1_std']:.3f}",
                            "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
                            "Landslides found": f"{v['landslides_found']} / {v['landslides_total']}",
                            "False alarms": v["false_alarms"]} for k, v in s.items()])
        c1, c2 = st.columns([1.15, 1])
        with c1:
            thr_key = next(k for k in s if k.startswith("Threshold") and "tuned" in k)
            st.markdown(f"<span style='font-size:20px'>U-Net — **{s['U-Net']['f1_mean']:.2f}**, "
                        f"coherence threshold — **{s[thr_key]['f1_mean']:.2f}**</span> (F1, mean over "
                        f"{len(bench['test_seeds'])} scenes)", unsafe_allow_html=True)
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption(f"Train scenes: seeds 0–{len(bench['train_seeds']) - 1}; test scenes: seeds "
                       f"{bench['test_seeds'][0]}–{bench['test_seeds'][-1]} (never seen in training). "
                       f"Thresholds were tuned on the training scenes "
                       f"(Δγ = {bench['tuned_threshold']}, |d| = "
                       f"{bench.get('displacement_threshold', '—')} mm). All methods share the same "
                       "post-processing.")
        with c2:
            rows = pd.DataFrame(bench["rows"])
            fig = px.strip(rows, x="method", y="f1", color="method",
                           color_discrete_sequence=["#e8590c", "#7f8c99", "#a7b1bc", "#4c9be8"])
            fig.add_trace(go.Scatter(x=list(s), y=[v["f1_mean"] for v in s.values()],
                                     mode="markers", marker=dict(symbol="line-ew-open", size=40,
                                                                 color="#fff"), name="mean"))
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="F1 per scene",
                              yaxis_range=[0, 1])
            st.plotly_chart(style(fig, 340), width="stretch")
        st.info("**Why the threshold loses:** in the Almaty scenes coherence also drops on seasonal "
                "snow (2700–3700 m) and ploughed fields on the plain, which do not move. Slow "
                "landslides keep their coherence at L-band but do move. A single Δγ cut can't "
                "separate these cases; the U-Net sees coherence *and* displacement together. "
                "**Caveat:** these are simulated scenes, so real-world scores will be lower. "
                "Validating against a landslide inventory is the next step.")
        if bench.get("real"):
            rb = bench["real"]
            rs = rb["summary"]
            r_thr = next(k for k in rs if k.startswith("Threshold") and "tuned" in k)
            st.markdown("#### Benchmark on real NISAR background, held-out dates")
            st.markdown(f"<span style='font-size:20px'>U-Net — **{rs['U-Net']['f1_mean']:.2f}**, "
                        f"coherence threshold — **{rs[r_thr]['f1_mean']:.2f}**</span> (F1, mean over "
                        f"{rb['n_test_scenes']} scenes)", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "Method": k, "F1": f"{v['f1_mean']:.3f} ± {v['f1_std']:.3f}",
                "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
                "Landslides found": f"{v['landslides_found']} / {v['landslides_total']}",
                "False alarms": v["false_alarms"]} for k, v in rs.items()]),
                hide_index=True, width="stretch")
            st.caption(f"Background = real NISAR GUNW pairs over Almaty (real atmosphere, snow, "
                       f"fields, noise) with landslides injected at known places. Trained on "
                       f"{', '.join(rb['train_pairs'])}; tested on {', '.join(rb['test_pairs'])} — "
                       f"dates never seen in training. Thresholds tuned on the training pairs "
                       f"(Δγ = {rb['tuned_threshold']}, |d| = {rb['displacement_threshold']} mm). "
                       "A real, unlabelled landslide in a test pair would count as a false alarm, "
                       "so these scores are a lower bound.")
            so = rb.get("synthetic_only_unet")
            if so:
                st.markdown(f"**Why we added real data:** the same U-Net trained on simulated scenes "
                            f"only scores F1 **{so['f1_mean']:.2f}** on these held-out real dates "
                            f"(found {so['landslides_found']}/{so['landslides_total']}) — real "
                            f"atmosphere and noise fool it. With real NISAR pairs in training: "
                            f"F1 **{rs['U-Net']['f1_mean']:.2f}** "
                            f"(found {rs['U-Net']['landslides_found']}/{rs['U-Net']['landslides_total']}).")
    else:
        st.info("No benchmark yet — run **Retrain + benchmark** in the sidebar, or "
                "`python scripts/train_and_benchmark.py`.")

# --- Forecast ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sim_stack(seed: int):
    stck = SARPhysics.generate_synthetic_stack(seed=seed, dem=get_dem())
    return stck


with tabs[4]:
    st.markdown("#### Will it fail, and when? Inverse-velocity forecast")
    st.markdown("<div class='note'>Before failure a slope accelerates, and 1 / velocity falls "
                "roughly linearly to zero (Fukuzono 1985). We stack consecutive 12-day NISAR "
                "pairs, follow each moving zone through time, and extrapolate the 1/v line to "
                "zero. That turns detection (\"it moved\") into warning (\"it may fail around this "
                "date\").</div>", unsafe_allow_html=True)
    chain = real_chain(items_all)
    opts = ["Simulated series (15 dates, Almaty terrain)"] + (
        [f"Real NISAR series ({len(chain) + 1} dates, track {REAL_TRACK})"] if len(chain) >= 3 else [])
    c1, c2, c3 = st.columns([2, 0.7, 1])
    fsrc = c1.radio("Time series", opts, horizontal=True)
    zthr = c3.slider("Min cumulative movement (mm)", 10, 60, 30, 5,
                     help="Below ~30 mm, accumulated atmospheric noise starts to look like "
                          "movement on the simulated series")
    if fsrc.startswith("Simulated"):
        fseed = c2.number_input("Series seed", 0, 999, 7)
        stck = sim_stack(int(fseed))
        dates, cohs, truth_z = stck["dates"], stck["coherences"], stck["zones"]
        cum = cumulative_displacement(stck["phase_increments"], cohs, stck["wavelength"],
                                      inc_angle, dem=get_dem())
    else:
        dates = [chain[0]["ref_date"]] + [it["sec_date"] for it in chain]
        cohs, truth_z = [it["coherence"] for it in chain], None
        cum = cumulative_displacement([it["unwrapped_phase"] for it in chain], cohs,
                                      float(chain[0]["wavelength"]), inc_angle, dem=get_dem())
    zones = moving_zones(cum, cohs, thr_mm=zthr)

    days = [(d - dates[0]).days for d in dates]
    _, acc = acceleration_map(cum, days)
    lat_min, lat_max, lon_min, lon_max = AOI
    h, w = cum.shape[1:]
    rows, results = [], []
    for i, zm in enumerate(zones):
        series = cum[:, zm].mean(1)
        res = inverse_velocity(dates, series)
        cy, cx = np.argwhere(zm).mean(0)
        lat = lat_max - cy / (h - 1) * (lat_max - lat_min)
        lon = lon_min + cx / (w - 1) * (lon_max - lon_min)
        row = {"Zone": f"Z-{i + 1:02d}", "Lat, Lon": f"{lat:.3f}, {lon:.3f}",
               "Total disp. (mm)": round(float(series[-1]), 1),
               "Velocity now (mm/day)": round(res["v_last"], 2), "Trend": res["status"],
               "Predicted failure": (f"{res['t_f_date']:%d %b %Y} "
                                     f"({res['t_f_low']:%d %b}–{res['t_f_high']:%d %b})"
                                     if res["status"] == "ACCELERATING" else "—"),
               "Warning": warning_level(res)}
        if truth_z is not None:
            match = next((z for z in truth_z if zm[int(z["cy"]), int(z["cx"])]), None)
            row["Simulated truth"] = (match["kind"] + (f", fails {match['t_f_date']}"
                                                       if match and "t_f_date" in match else "")
                                      if match else "noise")
        rows.append(row)
        results.append((row["Zone"], series, res))

    a, b, c, d = st.columns(4)
    n_acc = sum(r["Trend"] == "ACCELERATING" for r in rows)
    a.metric("Acquisitions", len(dates))
    b.metric("Moving zones", len(rows))
    c.metric("Accelerating", n_acc)
    soonest = min((r[2]["t_f_date"] for r in results if r[2]["status"] == "ACCELERATING"),
                  default=None)
    d.metric("Earliest predicted failure", f"{soonest:%d %b %Y}" if soonest else "none")

    c1, c2 = st.columns(2)
    v_mean = gaussian_filter(-(cum[-1] - cum[0]) / max(days[-1], 1), 1.5)
    acc_s = gaussian_filter(-acc, 2.5)
    outlines = np.zeros(v_mean.shape, bool)
    for zm in zones:
        outlines |= zm ^ (gaussian_filter(zm.astype(float), 1) > 0.9)
    for col, (arr, t, cm, unit) in zip((c1, c2), (
            (v_mean, "Mean velocity (+ = subsiding)", "RdBu_r", "mm/day"),
            (acc_s, "Acceleration (+ = speeding up)", "RdBu_r", "mm/day²"))):
        lim = float(np.percentile(np.abs(arr), 99.7)) or 1e-3
        fig = image_fig(np.where(outlines, np.nan, arr), t, cm, -lim, lim, unit, height=400)
        col.plotly_chart(fig, width="stretch")
    st.caption("Zone outlines are left blank in both maps. Positive acceleration in red is what "
               "an early-warning service should watch.")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.style.map(lambda v: f"color:{WARN_HEX.get(v, '')}", subset=["Warning"])
                     .format(precision=1, subset=["Total disp. (mm)"])
                     .format(precision=2, subset=["Velocity now (mm/day)"]),
                     hide_index=True, width="stretch")
        pick = st.selectbox("Zone to inspect", [r[0] for r in results],
                            index=next((i for i, r in enumerate(results)
                                        if r[2]["status"] == "ACCELERATING"), 0))
        _, series, res = next(r for r in results if r[0] == pick)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=dates, y=series, mode="lines+markers",
                                       line=dict(color="#e8590c", width=2.5)))
            fig.update_layout(yaxis_title="Cumulative vertical displacement (mm)")
            st.plotly_chart(style(fig, 330, f"{pick}: displacement"), width="stretch")
        with c2:
            from datetime import timedelta
            tm = [dates[0] + timedelta(days=float(t)) for t in res["t_mid"]]
            fig = go.Figure(go.Scatter(x=tm, y=res["inv_velocity"], mode="markers",
                                       marker=dict(color="#cfd6df", size=8), name="1 / v"))
            if res["status"] == "ACCELERATING":
                t_end = res["t_f"] + 2
                tt = np.linspace(res["fit_t"][0], t_end, 50)
                fig.add_trace(go.Scatter(x=[dates[0] + timedelta(days=float(t)) for t in tt],
                                         y=res["slope"] * tt + res["intercept"], mode="lines",
                                         line=dict(color="#d62828", dash="dash"), name="fit"))
                fig.add_vrect(x0=res["t_f_low"], x1=res["t_f_high"], fillcolor="#d62828",
                              opacity=0.15, line_width=0)
                fig.add_vline(x=res["t_f_date"], line_color="#d62828")
                fig.add_annotation(x=res["t_f_date"], y=max(res["inv_velocity"]) * 0.9,
                                   text=f"t_f ≈ {res['t_f_date']:%d %b}", showarrow=False,
                                   font=dict(color="#ff8a80"), xanchor="right")
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig.update_layout(yaxis_title="1 / velocity (day/mm)", showlegend=False,
                              yaxis_rangemode="tozero")
            st.plotly_chart(style(fig, 330, f"{pick}: inverse velocity"), width="stretch")
        st.caption("Warning levels we propose: ALERT = failure predicted within 14 days of the last "
                   "acquisition, WARNING = within 45 days, WATCH = moving > 1 mm/day or a distant "
                   "forecast. The window comes from refitting with 3–7 points.")
    else:
        st.success(f"No moving zones above {zthr} mm cumulative displacement.")

# --- Real NISAR data ---------------------------------------------------------
with tabs[5]:
    granules = asf_granules()
    chain_asf = consecutive_series(granules, REAL_TRACK, REAL_FRAME) if granules else []
    st.markdown("#### Real NISAR GUNW products over Almaty")
    a, b, c, d = st.columns(4)
    a.metric("GUNW granules over AOI", len(granules))
    b.metric("Tracks / frames", len({(g["track"], g["frame"]) for g in granules}))
    c.metric(f"Consecutive pairs, track {REAL_TRACK}", len(chain_asf))
    d.metric("Subsets downloaded", len(items_all))

    c1, c2 = st.columns([1, 1.25])
    with c1:
        fps = frame_footprints(granules)
        fig = go.Figure()
        for f in fps:
            xy = np.asarray(f["coords"])
            fig.add_trace(go.Scattermap(lon=xy[:, 0], lat=xy[:, 1], mode="lines",
                                        line=dict(width=1.5, color="#4c9be8" if (
                                            f["track"], f["frame"]) != (REAL_TRACK, REAL_FRAME)
                                            else "#e8590c"),
                                        name=f"T{f['track']}/F{f['frame']} · {f['coverage']}%",
                                        hoverinfo="name"))
        la0, la1, lo0, lo1 = AOI
        fig.add_trace(go.Scattermap(lon=[lo0, lo1, lo1, lo0, lo0], lat=[la0, la0, la1, la1, la0],
                                    mode="lines", line=dict(color="#fff", width=2), name="AOI"))
        fig.add_trace(go.Scattermap(lon=[s["lng"] for s in SITES], lat=[s["lat"] for s in SITES],
                                    mode="markers", marker=dict(size=9, color="#fff"),
                                    text=[s["name"] for s in SITES], name="Known sites"))
        fig.update_layout(map=dict(style="carto-darkmatter", center=dict(lat=43.2, lon=76.9),
                                   zoom=5.2), legend=dict(font=dict(size=10)))
        st.plotly_chart(style(fig, 420, "Frames intersecting the study area (% AOI coverage)"),
                        width="stretch")
        st.caption(f"We use descending track {REAL_TRACK}, frame {REAL_FRAME}: it covers 100% "
                   "of the AOI and has the longest unbroken 12-day chain. Track 127 / frame 23 "
                   "only clips the western ~45% of the AOI (Almaty is at its swath edge).")
    with c2:
        st.markdown(f"**Public NISAR browse images: unwrapped phase over the AOI, "
                    f"track {REAL_TRACK}**")
        crops = []
        for g in chain_asf:
            try:
                im = browse_crop(g, AOI, os.path.join(REAL_DIR, "browse"))
            except Exception:
                im = None
            if im is not None:
                crops.append((g, im))
        if crops:
            gcols = st.columns(5)
            for i, (g, im) in enumerate(crops):
                gcols[i % 5].image(im, caption=f"{g['ref_date']:%d.%m} → {g['sec_date']:%d.%m}",
                                   width="stretch")
            st.caption("These are real NISAR data (ASF browse PNGs, colour = phase). The city is "
                       "in the top half, the mountains in the noisy bottom half. The broad bands "
                       "that follow the mountain front are atmospheric delay, which is why we "
                       "high-pass the displacement before detection.")
        else:
            st.caption("Browse images unavailable offline.")

    st.divider()
    if items_all:
        st.markdown("#### Model check at known landslide sites")
        if source != SRC_REAL:
            st.info("Switch the sidebar data source to **Real NISAR · Almaty** to run the model "
                    "on the downloaded pairs.")
        else:
            st.dataframe(site_check(R, data), hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            hp = SARPhysics.highpass(R["vert"])
            for col, (arr, t, cm, lo, hi, u) in zip((c1, c2), (
                    (hp, "Vertical displacement (real)", "RdBu", -30, 30, "mm"),
                    (R["prob"], "U-Net probability (real)", "Viridis", 0, 1, ""))):
                fig = image_fig(np.where(R["valid"], arr, np.nan), t, cm, lo, hi, u, height=420)
                hh, ww = arr.shape
                fig.add_trace(go.Scatter(
                    x=[(s["lng"] - AOI[2]) / (AOI[3] - AOI[2]) * (ww - 1) for s in SITES],
                    y=[(AOI[1] - s["lat"]) / (AOI[1] - AOI[0]) * (hh - 1) for s in SITES],
                    mode="markers+text", text=[s["name"] for s in SITES],
                    textposition="top center", marker=dict(symbol="diamond-open", size=12,
                                                           color="#fff"), showlegend=False))
                col.plotly_chart(fig, width="stretch")
            st.caption("No ground truth exists for these dates. We read the model's response "
                       "at slopes where landslides are documented, and compare it with stable "
                       "ground elsewhere. Site coordinates are approximate.")
    else:
        st.markdown("#### Get the real interferograms")
        st.markdown(
            "Search and browse images are public. Downloading GUNW files requires a **free NASA "
            "Earthdata account**. Our loader never downloads the 2.5 GB file: it opens the HDF5 "
            "over HTTP range requests and reads only the chunks covering Almaty (a few MB per "
            "pair), applies the ionospheric phase screen, and resamples to the app's grid.")
        st.code("# 1. register at https://urs.earthdata.nasa.gov (free), then\n"
                "export EARTHDATA_USERNAME=...  EARTHDATA_PASSWORD=...\n"
                "# 2. fetch the Almaty subsets for the whole track-63 time series\n"
                "python scripts/fetch_almaty.py", language="bash")
        st.caption("Credentials found ✓ — run the script." if have_credentials()
                   else "No Earthdata credentials on this machine yet.")

# --- Method & Team ----------------------------------------------------------
with tabs[6]:
    st.markdown("#### Pipeline")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1b222c", color="#3a4654", fontcolor="#dde3ea", fontname="Helvetica", fontsize=11];
  edge [color="#6b7785"];
  asf [label="ASF search\\n46 GUNW over Almaty"]; sub [label="HTTP range read\\nAOI subset only"];
  iono [label="Ionosphere\\ncorrection"]; disp [label="Phase → LOS → vertical\\nhigh-pass 1.6 km"];
  coh [label="Coherence\\nprevious vs current pair"]; unet [label="U-Net (4 ch)\\n+ TTA ×4"];
  ev [label="Events, risk score,\\nGeoJSON"]; ts [label="Stack of 12-day pairs\\ncumulative displacement"];
  fc [label="Inverse velocity\\nfailure-time window"]; out [label="Globe · 3D terrain ·\\nwarning levels"];
  asf -> sub -> iono -> disp -> unet; sub -> coh -> unet; unet -> ev -> out; disp -> ts -> fc -> out;
}""")
    st.markdown("#### Decisions we made, and why")
    decisions = [
        ("Study area: Almaty foothills",
         "In 2024 a landslide on the Kamenskoe plateau killed a family of four, and the Medeu "
         "district monitors 85 landslide-prone sites. Cottages keep climbing the slopes. L-band "
         "(24 cm) keeps coherence on vegetated slopes, where C-band Sentinel-1 often fails."),
        ("Track 63 / frame 67, not 127 / 23",
         "Of the 46 granules over Almaty, 127/23 only clips ~45% of the AOI (swath edge). 63/67 "
         "covers 100% and has the longest unbroken 12-day chain, and the forecast needs that chain."),
        ("Read only the AOI out of each 2.5 GB file",
         "HDF5 is chunked, so HTTP range requests fetch a few MB. Downloading a whole time series "
         "would mean ~25 GB."),
        ("Train on simulation, on the real DEM",
         "No labelled landslide inventory exists yet for NISAR. We simulate on Almaty's real relief "
         "and include the failure modes we expect: snow decorrelation above 2700 m, fields on the "
         "plain, slow landslides that stay coherent, and topography-correlated atmosphere."),
        ("Coherence and displacement together",
         "Coherence alone confuses snow with landslides, and displacement alone confuses "
         "atmosphere with movement. A 4th input channel (high-passed displacement) lets the "
         "network require both."),
        ("A fair baseline",
         "Held-out test scenes, thresholds tuned on training scenes, identical post-processing, "
         "and object metrics as well as pixel F1. A missed slope matters more than a few "
         "boundary pixels."),
        ("Inverse velocity for the forecast",
         "It is simple, explainable to an emergency service, and has a long track record for "
         "slope failures. We report a date window, not a single date."),
    ]
    st.markdown("<table class='plain'><tr><th style='width:26%'>Decision</th><th>Reasoning</th>"
                "</tr>" + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in decisions)
                + "</table>", unsafe_allow_html=True)
    st.markdown("#### Limitations we know about")
    st.markdown(
        "- Benchmark scores are on **simulated** scenes; real accuracy will be lower until we "
        "validate against a landslide inventory (Kazselezashchita / akimat data).\n"
        "- One viewing geometry gives line-of-sight only. The vertical projection assumes "
        "vertical motion, but landslides move downslope. Combining ascending track 26 with "
        "descending track 63 (both cover the AOI) would give 2-D motion.\n"
        "- GUNW has no amplitude layer, so for real data the amplitude channel is modelled from "
        "terrain.\n"
        "- Inverse velocity works for brittle, accelerating failures. It cannot predict "
        "earthquake- or rain-triggered sudden collapse.\n"
        "- Known-site coordinates are approximate.")
    st.divider()
    st.markdown("#### Team")
    tcols = st.columns(len(TEAM))
    for col, name in zip(tcols, TEAM):
        col.markdown(f"<div style='border:1px solid rgba(255,255,255,0.1);border-radius:6px;"
                     f"padding:16px;text-align:center;font-size:17px'>{name}</div>",
                     unsafe_allow_html=True)
    st.markdown("<p class='note' style='margin-top:18px'>NASA Space Apps Challenge 2026 · "
                "Dancing with the SARs</p>", unsafe_allow_html=True)
