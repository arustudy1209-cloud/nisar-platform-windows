"""Model input features — one place, so synthetic and real NISAR data go through the
exact same preparation.

Channels:
  0  coherence before (previous 12-day pair)
  1  coherence after  (current pair)
  2  SAR amplitude, normalised
  3  high-passed vertical displacement, squashed with tanh(d / 30 mm)

Channel 3 is the reason the U-Net can beat a coherence threshold: it lets the network
require *movement* in addition to decorrelation (snow, fields) and catch slow landslides
that move but stay coherent.
"""

from __future__ import annotations

import numpy as np

from .physics import SARPhysics

N_CHANNELS = 4
DISP_SCALE_MM = 30.0


def build_input(coh_before, coh_after, amplitude, vert_mm) -> np.ndarray:
    disp = np.tanh(SARPhysics.highpass(vert_mm) / DISP_SCALE_MM)
    return np.stack([coh_before, coh_after, amplitude, disp]).astype(np.float32)


def scene_input(scene: dict, wavelength: float, inc_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """(features, vertical displacement mm) for a scene dict."""
    los = SARPhysics.phase_to_displacement(scene["unwrapped_phase"], wavelength)
    vert = SARPhysics.compute_los_to_vertical(los, inc_angle)
    x = build_input(scene["coherence_before"], scene["coherence_after"], scene["amplitude"], vert)
    return x, vert
