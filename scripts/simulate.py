"""Synthetic leakage sweep: zero group effect, growing subject fingerprint.

x_ij = b_i + e_ij,  b_i ~ N(0, s_b^2 I),  e_ij ~ N(0, s_w^2 I),  label y_i independent of x.
Writes results/synthetic.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adhd_eeg_audit import leakage

ROOT = Path(__file__).resolve().parents[1]
RATIOS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]   # s_b / s_w
MODELS = ["LogReg", "SVM-RBF", "5-NN"]
SEEDS = range(5)


def main() -> None:
    out = {"ratios_between_over_within_sd": RATIOS, "n_subjects": 60, "n_epochs": 30,
           "n_features": 20, "group_effect": 0.0, "results": {}}
    for name in MODELS:
        for scheme in ["epoch", "subject"]:
            curve = []
            for r in RATIOS:
                accs = []
                for s in SEEDS:
                    X, y, g = leakage.simulate_null(60, 30, 20, between_sd=r, within_sd=1.0, seed=s)
                    accs.append(leakage.cross_validate(leakage.make_models(s)[name], X, y, g,
                                                       scheme, seed=s)["epoch_bacc"])
                curve.append([float(np.mean(accs)), float(np.std(accs, ddof=1))])
            out["results"][f"{name}|{scheme}"] = curve
            print(name, scheme, [round(c[0], 3) for c in curve], flush=True)
    (ROOT / "results" / "synthetic.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
