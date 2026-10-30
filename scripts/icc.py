"""Intraclass correlation of the epoch features (how 'subject-specific' they are).

ICC(1) per feature from a one-way random-effects ANOVA (subjects as groups),
plus the design effect 1 + (m - 1) * ICC and the effective sample size.
Reads results/features.npz (written by run_audit.py); writes results/icc.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adhd_eeg_audit.stats import icc1

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    z = np.load(ROOT / "results" / "features.npz")
    X, g = z["X"].astype(np.float64), z["g"]
    iccs = np.array([icc1(X[:, j], g) for j in range(X.shape[1])])
    m = float(np.mean(np.bincount(g)))
    med = float(np.median(iccs))
    out = {"icc_median": med, "icc_q25": float(np.quantile(iccs, 0.25)),
           "icc_q75": float(np.quantile(iccs, 0.75)), "icc_min": float(iccs.min()),
           "icc_max": float(iccs.max()), "mean_epochs_per_subject": m,
           "design_effect_at_median_icc": 1 + (m - 1) * med,
           "n_epochs": int(len(g)),
           "effective_n_at_median_icc": float(len(g) / (1 + (m - 1) * med)),
           "icc_logTBR_Fz": float(iccs[-2]), "icc_logTBR_Cz": float(iccs[-1])}
    (ROOT / "results" / "icc.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
