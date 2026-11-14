"""SAR interferometry physics and the synthetic scene simulator.

The simulator is deliberately *not* friendly to a simple coherence threshold. Real L-band
interferograms over the Tien Shan foothills contain things that lose coherence without
moving (seasonal snow above ~2800 m, ploughed fields on the plain) and things that move
without losing much coherence (slow, coherent landslides — L-band keeps coherence on
vegetated slopes). Plus atmospheric phase that correlates with topography. Ground truth
marks only real slope movement.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from scipy.ndimage import gaussian_filter

NISAR_L_WAVELENGTH = 0.2384  # m, NISAR L-band centre frequency 1.257 GHz
REPEAT_DAYS = 12


class SARPhysics:
    """Physical conversions for L-band (NISAR) interferometric products."""

    def __init__(self, wavelength: float = NISAR_L_WAVELENGTH, incidence_angle_deg: float = 35.0):
        self.wavelength = wavelength
        self.incidence_angle_deg = incidence_angle_deg

    @staticmethod
    def phase_to_displacement(phase_array, wavelength: float = NISAR_L_WAVELENGTH):
        """Unwrapped phase (rad) -> line-of-sight displacement (mm)."""
        phase = np.asarray(phase_array, dtype=np.float64)
        return (phase * wavelength) / (4 * np.pi) * 1000.0

    @staticmethod
    def displacement_to_phase(disp_mm, wavelength: float = NISAR_L_WAVELENGTH):
        return np.asarray(disp_mm, dtype=np.float64) / 1000.0 * 4 * np.pi / wavelength

    @staticmethod
    def compute_los_to_vertical(los_mm, incidence_angle_deg: float = 35.0):
        """Project LOS displacement to vertical (assumes purely vertical motion)."""
        inc = np.deg2rad(incidence_angle_deg)
        return np.asarray(los_mm, dtype=np.float64) / np.cos(inc)

    @staticmethod
    def wrap_phase(phase):
        """Wrap phase into [-pi, pi)."""
        return (phase + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def highpass(disp, sigma: float = 20.0):
        """Remove long-wavelength signal (orbit ramp, large-scale atmosphere).

        Landslides in the study area are < 2 km across (< 25 px at 80 m posting), so a
        ~1.6 km Gaussian keeps them while dropping km-scale trends.
        """
        d = np.asarray(disp, dtype=np.float64)
        return d - gaussian_filter(d, sigma)

    # ------------------------------------------------------------------
    # Synthetic scenes
    # ------------------------------------------------------------------
    @staticmethod
    def _terrain(shape, rng, dem=None):
        h, w = shape
        if dem is None:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
            dem = (900 + 1800 * (1 - yy / h) ** 2
                   + 300 * np.sin(xx / w * 5 * np.pi) * (1 - yy / h)
                   + 60 * gaussian_filter(rng.normal(0, 1, shape), 6) * 10)
        dem = np.asarray(dem, dtype=np.float64)
        gy, gx = np.gradient(dem, 80.0)  # ~80 m posting
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        return dem, slope, gx, gy

    @staticmethod
    def _pick_centres(weight, n, rng, min_sep):
        p = weight.ravel() / weight.sum()
        h, w = weight.shape
        centres = []
        for _ in range(n):
            for _try in range(300):
                k = rng.choice(p.size, p=p)
                cy, cx = divmod(k, w)
                if 0.06 * h < cy < 0.94 * h and 0.06 * w < cx < 0.94 * w and \
                        all(np.hypot(cy - a, cx - b) > min_sep for a, b in centres):
                    centres.append((float(cy), float(cx)))
                    break
        return centres

    @staticmethod
    def _ellipse(xx, yy, cy, cx, sy, sx, theta):
        c, s = np.cos(theta), np.sin(theta)
        u = (xx - cx) * c + (yy - cy) * s
        v = -(xx - cx) * s + (yy - cy) * c
        return np.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))

    @staticmethod
    def _base_coherence(dem, slope, rng, shape):
        """Pre-event coherence: high on dry plains/rock, lower on steep vegetated slopes,
        very low on permanent snow/ice."""
        g = 0.88 - 0.006 * np.clip(slope - 10, 0, 40)
        g -= 0.45 * np.clip((dem - 3600) / 500, 0, 1)
        g += 0.04 * gaussian_filter(rng.normal(0, 1, shape), 3) * 3
        return np.clip(g, 0.12, 0.96)

    @staticmethod
    def _amplitude(slope, gx, rng, shape):
        # Slopes facing the (east-looking, ascending) radar are bright, back-slopes darker
        facing = np.clip(0.5 + 0.02 * np.degrees(np.arctan(-gx)), 0.05, 1.2)
        backscatter = 0.35 + 0.65 * facing * (0.6 + 0.4 * np.clip(slope / 35, 0, 1))
        amp = gaussian_filter(backscatter * rng.rayleigh(1.0, shape), 1.0)
        return (amp - amp.min()) / (np.ptp(amp) + 1e-9)

    @staticmethod
    def generate_synthetic_gunw(shape=(512, 512), seed: int = 42, dem=None,
                                wavelength: float = NISAR_L_WAVELENGTH):
        """One 12-day GUNW-like pair over the study area.

        Returns a dict with unwrapped_phase, wrapped_phase, coherence_before,
        coherence_after, amplitude, dem, truth_mask, confuser_mask, anomalies.
        """
        rng = np.random.default_rng(seed)
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        dem, slope, gx, gy = SARPhysics._terrain(shape, rng, dem)

        # --- background phase: orbit ramp + stratified + turbulent atmosphere
        phase = rng.uniform(-1, 1) * (xx / w) + rng.uniform(-1, 1) * (yy / h)
        phase += rng.uniform(-1.2, 1.2) * (dem - dem.mean()) / 1000.0
        phase += 1.5 * gaussian_filter(rng.normal(0, 1, shape), 18) * 18 / 3

        coh_before = SARPhysics._base_coherence(dem, slope, rng, shape)
        coh_after = coh_before - np.abs(0.035 * rng.normal(0, 1, shape))

        # --- landslides: foothill slopes 12-40 deg below the seasonal snow line
        weight = (np.clip((slope - 12) / 10, 0, 1) * np.clip((40 - slope) / 8, 0, 1)
                  * np.clip((2700 - dem) / 400, 0, 1)) + 1e-6
        n_slides = int(rng.integers(3, 7))
        truth = np.zeros(shape, dtype=bool)
        anomalies = []
        for cy, cx in SARPhysics._pick_centres(weight, n_slides, rng, 0.14 * min(h, w)):
            sy, sx = rng.uniform(3.5, 9), rng.uniform(5, 13)
            # Elongate downslope
            theta = float(np.arctan2(gy[int(cy), int(cx)], gx[int(cy), int(cx)]))
            g = SARPhysics._ellipse(xx, yy, cy, cx, sy, sx, theta)
            kind = "rapid" if rng.random() < 0.55 else "slow-coherent"
            disp_mm = -rng.uniform(25, 110) if kind == "rapid" else -rng.uniform(20, 60)
            drop = rng.uniform(0.32, 0.55) if kind == "rapid" else rng.uniform(0.05, 0.16)
            phase += SARPhysics.displacement_to_phase(disp_mm, wavelength) * g
            coh_after -= drop * g
            truth |= g > 0.35
            anomalies.append(dict(cy=cy, cx=cx, sy=sy, sx=sx, kind=kind,
                                  disp_mm=round(float(disp_mm), 1), coh_drop=round(float(drop), 2)))

        # --- confusers: decorrelation with NO movement
        confuser = np.zeros(shape, dtype=bool)
        snow_w = np.clip((dem - 2700) / 300, 0, 1) * np.clip((3700 - dem) / 300, 0, 1) + 1e-6
        field_w = np.clip((1000 - dem) / 150, 0, 1) * np.clip((6 - slope) / 3, 0, 1) + 1e-6
        for wgt, n, size, label in ((snow_w, int(rng.integers(1, 4)), (10, 26), "snow"),
                                    (field_w, int(rng.integers(1, 4)), (6, 14), "field")):
            if wgt.max() < 0.01:
                continue
            for cy, cx in SARPhysics._pick_centres(wgt, n, rng, 0.12 * min(h, w)):
                sy, sx = rng.uniform(*size), rng.uniform(*size)
                g = SARPhysics._ellipse(xx, yy, cy, cx, sy, sx, rng.uniform(0, np.pi))
                g = g * (0.7 + 0.3 * gaussian_filter(rng.random(shape), 2))
                coh_after -= rng.uniform(0.3, 0.5) * g
                confuser |= g > 0.35
                anomalies.append(dict(cy=cy, cx=cx, sy=sy, sx=sx, kind=f"confuser-{label}"))

        # Decorrelated pixels carry noisy phase
        coh_after = np.clip(gaussian_filter(coh_after, 1), 0.02, 1)
        phase += rng.normal(0, 1, shape) * np.sqrt(1 - coh_after ** 2) * 0.6

        return {
            "unwrapped_phase": phase.astype(np.float32),
            "wrapped_phase": SARPhysics.wrap_phase(phase).astype(np.float32),
            "coherence_before": coh_before.astype(np.float32),
            "coherence_after": coh_after.astype(np.float32),
            "amplitude": SARPhysics._amplitude(slope, gx, rng, shape).astype(np.float32),
            "dem": dem.astype(np.float32),
            "truth_mask": truth,
            "confuser_mask": confuser,
            "anomalies": anomalies,
        }

    @staticmethod
    def generate_synthetic_stack(shape=(512, 512), seed: int = 7, dem=None, n_dates: int = 15,
                                 end_date: date = date(2026, 10, 21),
                                 wavelength: float = NISAR_L_WAVELENGTH):
        """A time series of consecutive 12-day interferograms (like NISAR track 127,
        frame 23 over Almaty) with three slope behaviours:

        * accelerating — tertiary creep, 1/velocity falls linearly to zero at t_f
          (Fukuzono 1985), failure set a few days after the last acquisition
        * steady — constant creep
        * decelerating — movement dying out after a wet season
        """
        rng = np.random.default_rng(seed)
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        dem, slope, gx, gy = SARPhysics._terrain(shape, rng, dem)
        dates = [end_date - timedelta(days=REPEAT_DAYS * (n_dates - 1 - i)) for i in range(n_dates)]
        t = np.array([(d - dates[0]).days for d in dates], dtype=np.float64)

        weight = (np.clip((slope - 12) / 10, 0, 1) * np.clip((40 - slope) / 8, 0, 1)
                  * np.clip((2700 - dem) / 400, 0, 1)) + 1e-6
        behaviours = ["accelerating", "steady", "decelerating", "accelerating", "steady"]
        zones = []
        cum_true = np.zeros((n_dates, h, w))
        for (cy, cx), kind in zip(SARPhysics._pick_centres(weight, 5, rng, 0.16 * min(h, w)),
                                  behaviours):
            theta = float(np.arctan2(gy[int(cy), int(cx)], gx[int(cy), int(cx)]))
            g = SARPhysics._ellipse(xx, yy, cy, cx, rng.uniform(4, 8), rng.uniform(6, 12), theta)
            if kind == "accelerating":
                t_f = t[-1] + rng.uniform(5, 30)          # days after first acquisition
                v_last = rng.uniform(5, 12)                # mm/day at last acquisition
                A = 1.0 / (v_last * (t_f - t[-1]))
                d = -(1.0 / A) * -np.log(1 - t / t_f)      # integral of 1/(A (t_f - t))
                zone = dict(kind=kind, t_f_days=float(t_f),
                            t_f_date=(dates[0] + timedelta(days=float(t_f))).isoformat())
            elif kind == "steady":
                d = -rng.uniform(0.4, 1.5) * t
                zone = dict(kind=kind)
            else:
                v0, tau = rng.uniform(3, 6), rng.uniform(25, 50)
                d = -v0 * tau * (1 - np.exp(-t / tau))
                zone = dict(kind=kind)
            zone.update(cy=cy, cx=cx, mask=g > 0.4)
            zones.append(zone)
            cum_true += d[:, None, None] * g[None]

        base = SARPhysics._base_coherence(dem, slope, rng, shape)
        increments, coherences = [], []
        for k in range(1, n_dates):
            dd = cum_true[k] - cum_true[k - 1]
            ph = SARPhysics.displacement_to_phase(dd, wavelength)
            ph += rng.uniform(-0.8, 0.8) * (dem - dem.mean()) / 1000.0
            ph += 1.2 * gaussian_filter(rng.normal(0, 1, shape), 18) * 18 / 3
            # Faster movement -> more decorrelation
            speed = np.abs(dd) / REPEAT_DAYS
            coh = base - 0.04 * np.abs(rng.normal(0, 1, shape)) - np.clip(speed / 15, 0, 0.5)
            coh = np.clip(gaussian_filter(coh, 1), 0.02, 1)
            ph += rng.normal(0, 1, shape) * np.sqrt(1 - coh ** 2) * 0.5
            increments.append(ph.astype(np.float32))
            coherences.append(coh.astype(np.float32))

        return {
            "dates": dates,
            "phase_increments": increments,
            "coherences": coherences,
            "zones": zones,
            "dem": dem.astype(np.float32),
            "wavelength": wavelength,
        }
