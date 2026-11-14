"""Failure-time forecasting with the inverse-velocity method (Fukuzono 1985; Voight 1988).

A slope in tertiary creep accelerates so that 1/velocity decreases roughly linearly with
time. Extrapolating that line to 1/v = 0 gives the predicted failure time t_f.

What we do with a NISAR time series (consecutive 12-day pairs over one place):
  1. sum the per-pair displacements into a cumulative series per pixel
     (each pair is referenced to the median of stable, coherent ground first)
  2. average the series over each detected zone (single pixels are too noisy)
  3. velocity between acquisitions -> smooth over 3 points -> 1/v
  4. fit a line to the last few 1/v points; if the slope is clearly negative the zone
     is accelerating and t_f is where the line hits zero
  5. repeat the fit with different window lengths to get a spread for t_f — a forecast
     without an uncertainty window is not something we would hand to an emergency service
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from scipy.ndimage import gaussian_filter

from .physics import SARPhysics


def cumulative_displacement(phase_increments, coherences, wavelength, inc_angle=35.0,
                            coh_ref: float = 0.7, dem=None, smooth: float = 1.5):
    """(n_dates, H, W) cumulative vertical displacement in mm, first epoch = 0.

    Per pair: reference to stable coherent ground, remove the elevation-correlated
    (stratified tropospheric) delay by regressing against the DEM — 3.8 km of relief makes
    this the largest error term here — then high-pass and lightly smooth.
    """
    cum = [np.zeros_like(phase_increments[0], dtype=np.float64)]
    for ph, coh in zip(phase_increments, coherences):
        los = SARPhysics.phase_to_displacement(ph, wavelength)
        vert = SARPhysics.compute_los_to_vertical(los, inc_angle)
        stable = coh > coh_ref
        if stable.sum() < 100:
            stable = coh > np.percentile(coh, 50)
        if dem is not None:
            A = np.stack([np.ones(stable.sum()), dem[stable] / 1000.0], 1)
            (a, b), *_ = np.linalg.lstsq(A, vert[stable], rcond=None)
            vert = vert - (a + b * dem / 1000.0)
        else:
            vert = vert - np.median(vert[stable])
        d = SARPhysics.highpass(vert, sigma=25)
        if smooth:
            d = gaussian_filter(d, smooth)
        cum.append(cum[-1] + d)
    return np.stack(cum)


def moving_zones(cum, coherences, thr_mm: float = 30.0, min_px: int = 80,
                 min_coh: float = 0.4):
    """Connected regions whose cumulative movement exceeds thr_mm, ignoring pixels whose mean
    coherence is too low to trust the phase."""
    from scipy.ndimage import binary_fill_holes, binary_opening
    from skimage import measure

    mean_coh = np.mean(coherences, axis=0)
    m = (np.abs(cum[-1]) > thr_mm) & (mean_coh > min_coh)
    m = binary_fill_holes(binary_opening(m, iterations=1))
    lab = measure.label(m, connectivity=2)
    sizes = np.bincount(lab.ravel())
    return [lab == i for i in range(1, lab.max() + 1) if sizes[i] >= min_px]


def acceleration_map(cum, days):
    """Per-pixel quadratic fit d(t) = a + b t + c t^2. Returns (velocity_mm_per_day at the
    last epoch, acceleration_mm_per_day2) maps."""
    t = np.asarray(days, dtype=np.float64)
    A = np.stack([np.ones_like(t), t, t ** 2], axis=1)
    coef = np.linalg.pinv(A) @ cum.reshape(len(t), -1)
    b, c = coef[1], coef[2]
    v_last = (b + 2 * c * t[-1]).reshape(cum.shape[1:])
    return v_last, (2 * c).reshape(cum.shape[1:])


def _fit(t, y):
    A = np.stack([t, np.ones_like(t)], 1)
    (m, q), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([m, q])
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else 0.0
    return m, q, r2


def inverse_velocity(dates: list[date], disp_mm, window: int = 5, min_r2: float = 0.7):
    """Inverse-velocity analysis for one cumulative displacement series.

    Returns a dict with the velocity / 1/v series, trend status, and (if accelerating)
    the predicted failure date with a low-high window.
    """
    d = np.asarray(disp_mm, dtype=np.float64)
    t = np.array([(x - dates[0]).days for x in dates], dtype=np.float64)
    v = np.abs(np.diff(d)) / np.diff(t)                    # mm/day between acquisitions
    tm = (t[1:] + t[:-1]) / 2
    v_s = np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    v_s = np.maximum(v_s, 1e-3)
    inv = 1.0 / v_s

    n = len(inv)
    w = min(window, n)
    m, q, r2 = _fit(tm[-w:], inv[-w:])
    # Velocity trend over the same window, relative to the mean velocity
    mv, _, _ = _fit(tm[-w:], v_s[-w:])
    rel_acc = mv * (tm[-1] - tm[-w]) / max(v_s[-w:].mean(), 1e-6)

    out = dict(t_mid=tm, velocity=v_s, inv_velocity=inv, slope=m, intercept=q, r2=r2,
               v_last=float(v_s[-1]), window=w, fit_t=tm[-w:])
    if m < 0 and r2 >= min_r2 and rel_acc > 0.3:
        tf = -q / m
        spread = []
        for ww in range(max(3, w - 2), min(n, w + 2) + 1):
            mm, qq, rr = _fit(tm[-ww:], inv[-ww:])
            if mm < 0:
                spread.append(-qq / mm)
        lo, hi = (min(spread), max(spread)) if spread else (tf, tf)
        lo, hi = min(lo, tf), max(hi, tf)
        out.update(status="ACCELERATING", t_f=tf,
                   t_f_date=dates[0] + timedelta(days=float(tf)),
                   t_f_low=dates[0] + timedelta(days=float(lo)),
                   t_f_high=dates[0] + timedelta(days=float(hi)),
                   days_after_last=float(tf - t[-1]))
    elif rel_acc < -0.3:
        out.update(status="DECELERATING")
    else:
        out.update(status="STEADY")
    return out


def warning_level(res: dict) -> str:
    """Map a forecast to an operational level (what we would tell akimat / ЧС)."""
    if res["status"] != "ACCELERATING":
        return "WATCH" if res["v_last"] > 1.0 else "NORMAL"
    days = res["days_after_last"]
    if days <= 14:
        return "ALERT"
    if days <= 45:
        return "WARNING"
    return "WATCH"
