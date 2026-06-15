"""L3 硬件层指标 (FR-3.11/3.12) —— 真实现 (numpy 周期图, 无 scipy 依赖)。
FR-3.13 重复定位精度 / FR-3.14 摩擦辨识需要真机标定流程, 列入已知限制。"""
from __future__ import annotations

import numpy as np

from ..schema import Episode, Phase


def tracking_error(ep: Episode) -> dict:
    """FR-3.12 e_track: 实测 vs 指令的逐时刻偏差。
    - rms_m / peak_m: align→fasten 全窗口 (含运动段, 受跟踪迟滞影响)
    - steady_rms_m : fasten 窗口 (指令准静止) — 归因用的硬件判别量,
      运动迟滞不污染, 反映漂移/抖动/噪声。"""
    _, a0, _ = next(s for s in ep.robot.phase_spans if s[0] is Phase.ALIGN)
    _, f0, f1 = next(s for s in ep.robot.phase_spans if s[0] is Phase.FASTEN)
    diff = ep.robot.ee_xyz_actual[a0:f1] - ep.model.chunk.cmd_xyz[a0:f1]
    norms = np.linalg.norm(diff, axis=1)
    steady = norms[f0 - a0:]
    return {"rms_m": float(np.sqrt(np.mean(norms**2))),
            "peak_m": float(np.max(norms)),
            "steady_rms_m": float(np.sqrt(np.mean(steady**2)))}


def band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """周期图法频段功率。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    psd = (np.abs(X) ** 2) / (fs * len(x))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    df = freqs[1] - freqs[0]
    return float(psd[mask].sum() * df)


def jitter_band_power(ep: Episode, f_band=(5.0, 50.0)) -> float:
    """FR-3.11 末端抖动: 实测末端加速度 5–50 Hz 频段 PSD 能量 (三轴求和) [(m/s²)²]。"""
    t = ep.robot.t
    dt = float(t[1] - t[0])
    fs = 1.0 / dt
    acc = np.diff(ep.robot.ee_xyz_actual, n=2, axis=0) / dt**2
    return float(sum(band_power(acc[:, ax], fs, *f_band) for ax in range(3)))