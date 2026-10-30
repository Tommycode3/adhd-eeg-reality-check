"""Epoching, artifact rejection, Welch PSD and band-power features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

EPOCH_S = 4.0          # non-overlapping epochs
WELCH_SEG_S = 2.0      # Welch segment length -> 0.5 Hz resolution
TOTAL_BAND = (1.0, 40.0)
BANDS = {              # [lo, hi) in Hz
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
# TBR definitions. "narrow" follows the 13-21 Hz beta band used in much of the
# TBR literature (and in Arns et al., 2013); "broad" is the textbook 13-30 Hz.
TBR_DEFS = {"narrow": ((4.0, 8.0), (13.0, 21.0)), "broad": ((4.0, 8.0), (13.0, 30.0))}
REJECT_Z = 3.5


def epoch(x: np.ndarray, fs: float, epoch_s: float = EPOCH_S) -> np.ndarray:
    """(n_ch, n_samples) -> (n_epochs, n_ch, n_epoch_samples), tail dropped."""
    n = int(round(epoch_s * fs))
    k = x.shape[1] // n
    return x[:, : k * n].reshape(x.shape[0], k, n).transpose(1, 0, 2)


def epoch_psd(ep: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per epoch and channel: returns f, P (n_epochs, n_ch, n_freqs)."""
    nper = int(round(WELCH_SEG_S * fs))
    f, P = welch(ep, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                 detrend="constant", axis=-1)
    return f, P


def reject_mask(P: np.ndarray, f: np.ndarray, z: float = REJECT_Z) -> np.ndarray:
    """Robust, per-subject, unit-free artifact rejection.

    For every epoch and channel take log total power in TOTAL_BAND; compute a
    robust z-score across the subject's own epochs (median / 1.4826*MAD);
    keep the epoch if no channel exceeds ``z``. Unit-free on purpose: the
    amplitude scaling of the files is not documented.
    """
    sel = (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])
    lp = np.log(P[:, :, sel].sum(-1))                       # (n_ep, n_ch)
    med = np.median(lp, axis=0)
    mad = 1.4826 * np.median(np.abs(lp - med), axis=0) + 1e-12
    return (np.abs(lp - med) / mad).max(axis=1) <= z


def band_power(P: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sum of PSD bins in [lo, hi) times bin width (rectangle rule)."""
    df = f[1] - f[0]
    sel = (f >= lo) & (f < hi)
    return P[..., sel].sum(-1) * df


def tbr(P: np.ndarray, f: np.ndarray, kind: str = "narrow") -> np.ndarray:
    (tl, th), (bl, bh) = TBR_DEFS[kind]
    return band_power(P, f, tl, th) / band_power(P, f, bl, bh)


def relative_powers(P: np.ndarray, f: np.ndarray) -> np.ndarray:
    """(..., n_ch, n_f) -> (..., n_ch, n_bands)."""
    tot = band_power(P, f, *TOTAL_BAND)
    return np.stack([band_power(P, f, lo, hi) / tot for lo, hi in BANDS.values()], -1)


@dataclass
class SubjectFeatures:
    subject: str
    group: int
    n_epochs_total: int
    n_epochs_kept: int
    psd_mean: np.ndarray        # (n_ch, n_f) mean of kept epochs
    epoch_feats: np.ndarray     # (n_kept, n_feat) ML features


def feature_names(channels: list[str]) -> list[str]:
    names = [f"logrel_{b}_{c}" for c in channels for b in BANDS]
    return names + ["logTBR_Fz", "logTBR_Cz"]


def subject_features(x: np.ndarray, fs: float, channels: list[str], subject: str,
                     group: int, reject: bool = True) -> SubjectFeatures:
    ep = epoch(x, fs)
    f, P = epoch_psd(ep, fs)
    keep = reject_mask(P, f) if reject else np.ones(len(P), bool)
    P = P[keep]
    rel = relative_powers(P, f).reshape(len(P), -1)          # channel-major
    iz, icz = channels.index("Fz"), channels.index("Cz")
    t = tbr(P, f, "narrow")
    feats = np.column_stack([np.log10(rel), np.log10(t[:, iz]), np.log10(t[:, icz])])
    return SubjectFeatures(subject, group, len(keep), int(keep.sum()),
                           P.mean(0), feats.astype(np.float32))
