"""Render figures/*.png from results/*.json (no recomputation of statistics)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from adhd_eeg_audit import data, features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Palette validated with the dataviz validator (CVD-safe pair).
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY, INK, MUTED, SURF = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
C_CTRL, C_ADHD = BLUE, ORANGE
C_EPOCH, C_SUBJ = ORANGE, BLUE

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
})


def load(name):
    return json.loads((RES / name).read_text())


def fig_tbr():
    tb = load("tbr.json")
    rows = (RES / "tbr_per_subject.csv").read_text().splitlines()
    hdr = rows[0].split(",")
    grp = np.array([r.split(",")[1] == "ADHD" for r in rows[1:]])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, site in zip(axes, ["Cz", "Fz"]):
        key = f"TBR_narrow_{site}"
        col = hdr.index(key)
        v = np.array([float(r.split(",")[col]) for r in rows[1:]])
        prim = next(p for p in tb["primary"] if p["feature"] == key)
        for k, (mask, c, lab) in enumerate([(~grp, C_CTRL, "Control"), (grp, C_ADHD, "ADHD")]):
            x = v[mask]
            ax.scatter(k + rng.uniform(-0.18, 0.18, x.size), x, s=16, color=c, alpha=0.55,
                       edgecolor=SURF, linewidth=0.5, zorder=2)
            lx = np.log10(x)
            m = lx.mean()
            boot = np.array([rng.choice(lx, lx.size).mean() for _ in range(5000)])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            ax.errorbar(k + 0.32, 10 ** m, yerr=[[10 ** m - 10 ** lo], [10 ** hi - 10 ** m]],
                        fmt="o", color=INK, ms=6, capsize=4, lw=1.5, zorder=3)
            ax.text(k, 0.97, f"{lab}\nn={x.size}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=INK)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 40)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"θ/β ratio at {site}  (θ 4–8 Hz, β 13–21 Hz)")
        ax.text(0.5, 0.02, f"Hedges' g = {prim['g']:.2f}  [95% CI {prim['g_ci95'][0]:.2f}, "
                f"{prim['g_ci95'][1]:.2f}]\nAUC = {prim["auc"]:.2f}   Holm-adjusted permutation p (log TBR) = {prim["p_holm"]:.3f}",
                transform=ax.transAxes, ha="center", va="bottom", color=INK, fontsize=9)
    axes[0].set_ylabel("TBR (log scale)")
    fig.text(0.01, 0.005, "Dots: one child each. Black: geometric mean, 95% bootstrap CI.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig1_tbr_by_group.png", dpi=200)
    plt.close(fig)


def fig_cv():
    lk = load("leakage.json")
    models = ["LogReg", "SVM-RBF", "RandomForest", "5-NN"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    w = 0.26
    x = np.arange(len(models))
    series = [
        ("Epoch-level CV (leaky)", C_EPOCH, [lk["cv"][f"{m}|epoch"]["epoch_bacc_mean"] for m in models],
         [lk["cv"][f"{m}|epoch"]["epoch_bacc_sd"] for m in models], None),
        ("Epoch-level CV, diagnosis labels shuffled between subjects", "#f3b49b",
         [lk["label_shuffled_epoch_cv"][m]["mean"] for m in models],
         [np.std(lk["label_shuffled_epoch_cv"][m]["all"], ddof=1) for m in models], "///"),
        ("Subject-level CV (honest)", C_SUBJ, [lk["cv"][f"{m}|subject"]["epoch_bacc_mean"] for m in models],
         [lk["cv"][f"{m}|subject"]["epoch_bacc_sd"] for m in models], None),
    ]
    for k, (lab, c, mu, sd, hatch) in enumerate(series):
        xs = x + (k - 1) * w
        ax.bar(xs, mu, w - 0.03, color=c, label=lab, hatch=hatch, edgecolor=SURF if not hatch else ORANGE,
               linewidth=0.0 if not hatch else 0.0, zorder=2)
        ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=INK, capsize=3, lw=1, zorder=3)
        for xi, m, e in zip(xs, mu, sd):
            ax.text(xi, m + e + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(len(models) - 0.5, 0.505, "chance", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x, models)
    ax.set_ylim(0.3, 1.08)
    ax.set_ylabel("Balanced accuracy (epochs)")
    ax.set_title("Same features, same models: only the split changes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_epoch_vs_subject_cv.png", dpi=200)
    plt.close(fig)


def fig_subject_id():
    lk = load("leakage.json")
    sid = lk["subject_id"]
    models = ["LogReg", "RandomForest", "5-NN"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    vals = [sid[m] for m in models]
    ax.barh(models, vals, color=GREY, height=0.55, zorder=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", color=INK)
    ax.axvline(sid["chance"], color=ORANGE, lw=1.5)
    ax.text(sid["chance"] + 0.01, 2.45, f"chance = 1/121 = {sid['chance']:.1%}", color=INK, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 accuracy: which of 121 children produced this 4-s epoch?")
    ax.set_title("Each child's EEG is a fingerprint")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_subject_identification.png", dpi=200)
    plt.close(fig)


def fig_null():
    lk = load("leakage.json")
    nl = lk["subject_cv_null"]["LogReg"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(nl["null"], bins=25, color=GREY, zorder=2)
    ax.axvline(nl["observed_mean_over_seeds"], color=BLUE, lw=2)
    ax.text(nl["observed_mean_over_seeds"], ax.get_ylim()[1] * 0.92,
            f"  observed {nl['observed_mean_over_seeds']:.3f}\n  p = {nl['p']:.3f}", color=INK, fontsize=9,
            va="top")
    ax.set_xlabel("Subject-level CV balanced accuracy (LogReg)")
    ax.set_ylabel("Permutations")
    ax.set_title(f"Null: diagnosis shuffled between subjects ({nl['n_perm']} permutations)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_subject_cv_null.png", dpi=200)
    plt.close(fig)


def fig_synthetic():
    sy = load("synthetic.json")
    r = sy["ratios_between_over_within_sd"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    styles = {"LogReg": "o", "SVM-RBF": "s", "5-NN": "^"}
    for m, mk in styles.items():
        for scheme, c in [("epoch", C_EPOCH), ("subject", C_SUBJ)]:
            mu = np.array(sy["results"][f"{m}|{scheme}"])[:, 0]
            ax.plot(r, mu, marker=mk, color=c, lw=2, ms=7,
                    label=f"{m}, {'epoch' if scheme == 'epoch' else 'subject'}-level CV")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Subject fingerprint strength  σ_between / σ_within")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("Synthetic data with ZERO group effect")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(FIG / "fig5_synthetic_leakage.png", dpi=200)
    plt.close(fig)


def fig_relpower():
    tb = load("tbr.json")
    ch, bands = data.CHANNELS, list(features.BANDS)
    G = np.zeros((len(bands), len(ch)))
    Q = np.ones_like(G)
    for e in tb["exploratory_relpower"]:
        G[bands.index(e["band"]), ch.index(e["channel"])] = e["g"]
        Q[bands.index(e["band"]), ch.index(e["channel"])] = e["q_bh"]
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#ecebe7", ORANGE])
    lim = max(0.6, np.abs(G).max())
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    im = ax.imshow(G, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(bands)):
        for j in range(len(ch)):
            ax.text(j, i, f"{G[i, j]:.2f}" + ("*" if Q[i, j] < 0.05 else ""), ha="center", va="center",
                    fontsize=6.5, color=INK)
    ax.set_xticks(range(len(ch)), ch, fontsize=8)
    ax.set_yticks(range(len(bands)), bands)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("Hedges' g (ADHD − control)")
    ax.set_title("Log relative band power: exploratory, * = BH-FDR q < 0.05")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_relative_power_effects.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_tbr()
    fig_cv()
    fig_subject_id()
    fig_null()
    if (RES / "synthetic.json").exists():
        fig_synthetic()
    fig_relpower()
    print("figures written to", FIG)
