"""L3 hardware-layer metrics (FR-3.11/3.12) — real implementation (numpy periodogram, no scipy dependency).
FR-3.13 repeatability / FR-3.14 friction identification require a real-robot calibration procedure, listed as a known limitation."""
from __future__ import annotations

import numpy as np

from ..schema import Episode, Phase


def tracking_error(ep: Episode) -> dict:
    """FR-3.12 e_track: per-step deviation of actual vs command.
    - rms_m / peak_m: full align→fasten window (includes motion segments, affected by tracking lag)
    - steady_rms_m : fasten window (command quasi-static) — the hardware discriminator used for attribution,
      uncontaminated by motion lag, reflects drift/jitter/noise."""
    _, a0, _ = next(s for s in ep.robot.phase_spans if s[0] is Phase.ALIGN)
    _, f0, f1 = next(s for s in ep.robot.phase_spans if s[0] is Phase.FASTEN)
    diff = ep.robot.ee_xyz_actual[a0:f1] - ep.model.chunk.cmd_xyz[a0:f1]
    norms = np.linalg.norm(diff, axis=1)
    steady = norms[f0 - a0:]
    return {"rms_m": float(np.sqrt(np.mean(norms**2))),
            "peak_m": float(np.max(norms)),
            "steady_rms_m": float(np.sqrt(np.mean(steady**2)))}


def band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Band power via the periodogram method."""
    x = np.asarray(x, float)
    x = x - x.mean()
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    psd = (np.abs(X) ** 2) / (fs * len(x))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    df = freqs[1] - freqs[0]
    return float(psd[mask].sum() * df)


def jitter_band_power(ep: Episode, f_band=(5.0, 50.0)) -> float:
    """FR-3.11 end-effector jitter: PSD energy of measured end-effector acceleration in the 5–50 Hz band (summed over 3 axes) [(m/s²)²]."""
    t = ep.robot.t
    dt = float(t[1] - t[0])
    fs = 1.0 / dt
    acc = np.diff(ep.robot.ee_xyz_actual, n=2, axis=0) / dt**2
    return float(sum(band_power(acc[:, ax], fs, *f_band) for ax in range(3)))