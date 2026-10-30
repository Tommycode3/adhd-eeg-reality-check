"""Reproduce every number in the README.

    python scripts/fetch_data.py        # once
    python scripts/run_audit.py         # ~10-20 min on a laptop, single core
    python scripts/make_figures.py

Writes results/*.json, results/*.csv. Single-threaded, one subject in memory at a time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch

from adhd_eeg_audit import data, features, leakage, stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
N_BOOT = 10_000
N_PERM = 10_000
SEEDS = [0, 1, 2, 3, 4]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ features
def extract() -> dict:
    cache = RES / "features.npz"   # derived, git-ignored
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    X, y, g, rows, psd = [], [], [], [], []
    lo_pow, alpha_pow, corr = np.zeros(19), np.zeros(19), np.zeros((19, 19))
    for i, (subj, grp, path) in enumerate(data.list_subjects()):
        rec = data.load_recording(path, grp)
        sf = features.subject_features(rec.x, data.FS, data.CHANNELS, subj, grp)
        X.append(sf.epoch_feats)
        y += [grp] * sf.n_epochs_kept
        g += [i] * sf.n_epochs_kept
        rows.append((i, grp, sf.n_epochs_total, sf.n_epochs_kept, rec.x.shape[1] / data.FS))
        psd.append(sf.psd_mean)
        # channel-order sanity statistics (whole recording)
        xc = rec.x - rec.x.mean(1, keepdims=True)
        f, P = welch(xc, fs=data.FS, nperseg=256)
        lo_pow += np.log(P[:, (f >= 0.5) & (f <= 3)].mean(1))
        alpha_pow += np.log(P[:, (f >= 8) & (f < 13)].mean(1))
        corr += np.corrcoef(xc)
        del rec, xc
        if i % 20 == 0:
            log(f"features {i + 1}/121")
    n = len(rows)
    out = dict(X=np.vstack(X), y=np.array(y), g=np.array(g), subj_rows=np.array(rows),
               psd=np.stack(psd), f=features.epoch_psd(np.zeros((1, 1, 512)), data.FS)[0],
               subj_names=np.array([s for s, _, _ in data.list_subjects()]),
               lo_pow=lo_pow / n, alpha_pow=alpha_pow / n, corr=corr / n)
    np.savez_compressed(cache, **out)
    return out


def channel_sanity(d: dict) -> dict:
    ch = data.CHANNELS
    lo = d["lo_pow"] - d["lo_pow"].mean()
    al = d["alpha_pow"] - d["alpha_pow"].mean()
    c = d["corr"].copy()
    np.fill_diagonal(c, -1)
    return {
        "top2_low_freq_0.5-3Hz_channels (expect Fp1,Fp2: blinks)": [ch[i] for i in np.argsort(-lo)[:2]],
        "nearest_neighbour_by_correlation": {ch[i]: ch[int(np.argmax(c[i]))] for i in range(19)},
        "alpha_rank_desc": [ch[i] for i in np.argsort(-al)],
    }


def recording_length(rows: np.ndarray) -> dict:
    """Task duration depends on response speed -> a potential group confound."""
    adhd = rows[:, 1] == 1
    out = {}
    for j, name in [(4, "duration_s"), (3, "epochs_kept")]:
        a, b = rows[adhd, j], rows[~adhd, j]
        out[name] = {"median_adhd": float(np.median(a)), "median_control": float(np.median(b)),
                     "g": float(stats.hedges_g(a, b)),
                     "g_ci95": stats.bootstrap_ci(stats.hedges_g, a, b, N_BOOT, seed=7),
                     "p_perm": stats.perm_test(a, b, N_PERM, seed=8)}
    return out


# ------------------------------------------------------------------ TBR
def tbr_analysis(d: dict) -> dict:
    psd, f = d["psd"], d["f"]
    grp = d["subj_rows"][:, 1].astype(int)
    ch = data.CHANNELS
    out = {"n_adhd": int(grp.sum()), "n_control": int((grp == 0).sum()),
           "mde_g_80pct_power": stats.mde_two_sample(int(grp.sum()), int((grp == 0).sum())),
           "primary": [], "per_subject": {}}
    tests = []
    for kind in ["narrow", "broad"]:
        t = features.tbr(psd, f, kind)        # (n_subj, n_ch)
        for site in ["Cz", "Fz"]:
            v = t[:, ch.index(site)]
            out["per_subject"][f"TBR_{kind}_{site}"] = v.tolist()
            a, b = v[grp == 1], v[grp == 0]
            la, lb = np.log10(a), np.log10(b)
            r = {
                "feature": f"TBR_{kind}_{site}",
                "beta_band_hz": features.TBR_DEFS[kind][1],
                "mean_adhd": float(a.mean()), "sd_adhd": float(a.std(ddof=1)),
                "mean_control": float(b.mean()), "sd_control": float(b.std(ddof=1)),
                "median_adhd": float(np.median(a)), "median_control": float(np.median(b)),
                "g": float(stats.hedges_g(a, b)),
                "g_ci95": stats.bootstrap_ci(stats.hedges_g, a, b, N_BOOT, seed=1),
                "g_log": float(stats.hedges_g(la, lb)),
                "g_log_ci95": stats.bootstrap_ci(stats.hedges_g, la, lb, N_BOOT, seed=2),
                "auc": stats.auc(a, b),
                "p_perm_log": stats.perm_test(la, lb, N_PERM, seed=3),
            }
            tests.append(r)
    padj = stats.holm([r["p_perm_log"] for r in tests])
    for r, p in zip(tests, padj):
        r["p_holm"] = float(p)
    out["primary"] = tests

    # exploratory: log relative power, 5 bands x 19 channels, BH-FDR
    rel = np.log10(features.relative_powers(psd, f))          # (n_subj, n_ch, n_band)
    expl = []
    for ci, c in enumerate(ch):
        for bi, bname in enumerate(features.BANDS):
            v = rel[:, ci, bi]
            expl.append({"channel": c, "band": bname,
                         "g": float(stats.hedges_g(v[grp == 1], v[grp == 0])),
                         "p_perm": stats.perm_test(v[grp == 1], v[grp == 0], 5000, seed=ci * 10 + bi)})
    q = stats.bh([e["p_perm"] for e in expl])
    for e, qq in zip(expl, q):
        e["q_bh"] = float(qq)
    out["exploratory_relpower"] = expl
    out["exploratory_n_q_lt_0.05"] = int((q < 0.05).sum())
    out["exploratory_max_abs_g"] = max(expl, key=lambda e: abs(e["g"]))
    return out


# ------------------------------------------------------------------ leakage
def leakage_analysis(d: dict) -> dict:
    X, y, g = d["X"].astype(np.float64), d["y"], d["g"]
    out = {"n_epochs": int(len(y)), "n_features": int(X.shape[1]),
           "epochs_per_subject_median": float(np.median(np.bincount(g))),
           "cv": {}, "label_shuffled_epoch_cv": {}, "subject_cv_null": {}, "subject_id": {}}
    for name in leakage.make_models():
        for scheme in ["epoch", "subject"]:
            rs = []
            for s in SEEDS:
                m = leakage.make_models(s)[name]
                rs.append(leakage.cross_validate(m, X, y, g, scheme, seed=s))
            out["cv"][f"{name}|{scheme}"] = {
                "epoch_bacc_mean": float(np.mean([r["epoch_bacc"] for r in rs])),
                "epoch_bacc_sd": float(np.std([r["epoch_bacc"] for r in rs], ddof=1)),
                "subject_bacc_mean": float(np.mean([r["subject_bacc"] for r in rs])),
                "subject_bacc_sd": float(np.std([r["subject_bacc"] for r in rs], ddof=1)),
                "per_seed": rs}
            log(f"{name:12s} {scheme:7s} epoch-bacc={out['cv'][f'{name}|{scheme}']['epoch_bacc_mean']:.3f}")

    # TBR-only model (2 features) with honest subject-level CV
    Xt = X[:, -2:]
    rs = [leakage.cross_validate(leakage.make_models(s)["LogReg"], Xt, y, g, "subject", seed=s) for s in SEEDS]
    out["tbr_only_logreg_subject_cv"] = {
        "epoch_bacc_mean": float(np.mean([r["epoch_bacc"] for r in rs])),
        "subject_bacc_mean": float(np.mean([r["subject_bacc"] for r in rs]))}

    # Diagnosis labels shuffled BETWEEN subjects: there is no ADHD signal left at all.
    rng = np.random.default_rng(123)
    perms = [leakage.permute_subject_labels(y, g, rng) for _ in range(10)]
    for name in leakage.make_models():
        accs = [leakage.cross_validate(leakage.make_models(k)[name], X, yp, g, "epoch", seed=k)["epoch_bacc"]
                for k, yp in enumerate(perms)]
        out["label_shuffled_epoch_cv"][name] = {"mean": float(np.mean(accs)), "min": float(np.min(accs)),
                                                "max": float(np.max(accs)), "all": accs}
        log(f"shuffled-label epoch-CV {name}: {np.mean(accs):.3f}")

    # Permutation null for the honest (subject-level) LogReg estimate.
    n_null = 200
    obs = out["cv"]["LogReg|subject"]["epoch_bacc_mean"]
    null = [leakage.cross_validate(leakage.make_models()["LogReg"], X,
                                   leakage.permute_subject_labels(y, g, rng), g, "subject", seed=k)["epoch_bacc"]
            for k in range(n_null)]
    out["subject_cv_null"]["LogReg"] = {
        "observed_mean_over_seeds": obs, "null_mean": float(np.mean(null)),
        "null_95pct": float(np.quantile(null, 0.95)),
        "p": float((1 + np.sum(np.array(null) >= obs)) / (n_null + 1)), "n_perm": n_null, "null": null}
    log(f"subject-CV null: obs={obs:.3f} p={out['subject_cv_null']['LogReg']['p']:.3f}")

    # Who is this? 121-way subject identification from single 4-s epochs.
    for name in ["LogReg", "RandomForest", "5-NN"]:
        out["subject_id"][name] = leakage.subject_id_accuracy(leakage.make_models()[name], X, g)
        log(f"subject-ID {name}: {out['subject_id'][name]:.3f}")
    out["subject_id"]["chance"] = 1 / len(np.unique(g))
    return out


def main() -> None:
    t0 = time.time()
    d = extract()
    rows = d["subj_rows"]
    summary = {
        "dataset": "Nasrabadi et al. 2020, IEEE DataPort, doi:10.21227/rzfh-zn36, CC BY 4.0",
        "n_subjects": int(len(rows)), "n_adhd": int(rows[:, 1].sum()),
        "recording_s_median": float(np.median(rows[:, 4])),
        "recording_s_range": [float(rows[:, 4].min()), float(rows[:, 4].max())],
        "epochs_total": int(rows[:, 2].sum()), "epochs_kept": int(rows[:, 3].sum()),
        "channel_sanity": channel_sanity(d),
        "recording_length_by_group": recording_length(rows),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    log("TBR analysis")
    tb = tbr_analysis(d)
    (RES / "tbr.json").write_text(json.dumps(tb, indent=2))
    with open(RES / "tbr_per_subject.csv", "w") as fh:
        keys = list(tb["per_subject"])
        fh.write("subject,group," + ",".join(keys) + "\n")
        for i, s in enumerate(d["subj_names"]):
            fh.write(f"{s},{'ADHD' if rows[i, 1] else 'control'}," +
                     ",".join(f"{tb['per_subject'][k][i]:.5f}" for k in keys) + "\n")
    for r in tb["primary"]:
        log(f"{r['feature']}: g={r['g']:.3f} CI={r['g_ci95']} g_log={r['g_log']:.3f} "
            f"AUC={r['auc']:.3f} p_holm={r['p_holm']:.4f}")
    log("leakage analysis")
    lk = leakage_analysis(d)
    (RES / "leakage.json").write_text(json.dumps(lk, indent=2))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
