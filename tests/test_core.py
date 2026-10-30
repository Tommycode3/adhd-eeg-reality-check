import numpy as np
import pytest
from scipy import stats as sps

from adhd_eeg_audit import data, features, leakage, stats


# ---------- the key verifier: leakage on pure-null synthetic data ----------

@pytest.mark.parametrize("model_name", ["RandomForest", "5-NN"])
def test_epoch_cv_inflates_under_null(model_name):
    """No group effect at all, only subject fingerprints:
    epoch-level CV must look 'excellent', subject-level CV must sit at chance."""
    ep, sub = [], []
    for seed in range(3):
        X, y, g = leakage.simulate_null(n_subjects=40, n_epochs=30, group_effect=0.0, seed=seed)
        m = leakage.make_models(seed)[model_name]
        ep.append(leakage.cross_validate(m, X, y, g, "epoch", seed=seed)["epoch_bacc"])
        sub.append(leakage.cross_validate(m, X, y, g, "subject", seed=seed)["epoch_bacc"])
    assert np.mean(ep) > 0.90, ep
    assert 0.30 < np.mean(sub) < 0.70, sub


def test_subject_identity_is_decodable_under_null():
    X, y, g = leakage.simulate_null(n_subjects=40, n_epochs=30, seed=1)
    acc = leakage.subject_id_accuracy(leakage.make_models()["LogReg"], X, g)
    assert acc > 0.9  # chance = 1/40


def test_subject_cv_detects_real_effect():
    """Sanity: subject-CV is not just 'always chance' - a true group effect is found."""
    X, y, g = leakage.simulate_null(n_subjects=60, group_effect=1.5, seed=2)
    r = leakage.cross_validate(leakage.make_models()["LogReg"], X, y, g, "subject", seed=2)
    assert r["subject_bacc"] > 0.8


def test_subject_cv_never_splits_a_subject():
    X, y, g = leakage.simulate_null(n_subjects=20, n_epochs=5)
    for tr, te in leakage.splits("subject", y, g, 5, 0):
        assert not set(g[tr]) & set(g[te])


def test_permute_subject_labels_keeps_subject_constant():
    X, y, g = leakage.simulate_null(n_subjects=20, n_epochs=5)
    yp = leakage.permute_subject_labels(y, g, np.random.default_rng(0))
    for s in np.unique(g):
        assert len(set(yp[g == s])) == 1
    assert yp.sum() == y.sum()


# ---------- statistics ----------

def test_hedges_g_matches_formula():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 30), rng.normal(0, 1, 25)
    sp = np.sqrt(((29) * a.var(ddof=1) + 24 * b.var(ddof=1)) / 53)
    d = (a.mean() - b.mean()) / sp
    assert stats.hedges_g(a, b) == pytest.approx(d * stats.hedges_j(53))
    assert stats.hedges_j(53) == pytest.approx(1 - 3 / (4 * 53 - 1), abs=1e-4)


def test_bootstrap_ci_covers_truth_roughly():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.5, 1, 200), rng.normal(0, 1, 200)
    lo, hi = stats.bootstrap_ci(stats.hedges_g, a, b, n_resamples=2000)
    assert lo < 0.5 < hi


def test_permutation_close_to_t_test():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0.4, 1, 60), rng.normal(0, 1, 60)
    p = stats.perm_test(a, b, n_resamples=5000)
    assert p == pytest.approx(sps.ttest_ind(a, b).pvalue, abs=0.02)


def test_holm_and_bh():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(stats.holm(p), [0.04, 0.09, 0.09, 0.5])
    assert np.allclose(stats.bh(p), [0.04, 0.0533333, 0.0533333, 0.5])


# ---------- features ----------

def test_tbr_on_synthetic_sines():
    fs = 128.0
    t = np.arange(int(60 * fs)) / fs
    rng = np.random.default_rng(0)
    # theta amplitude 2, beta amplitude 1 -> power ratio 4
    x = 2 * np.sin(2 * np.pi * 6 * t) + np.sin(2 * np.pi * 17 * t) + 0.01 * rng.normal(size=t.size)
    ep = features.epoch(x[None], fs)
    f, P = features.epoch_psd(ep, fs)
    assert np.median(features.tbr(P, f, "narrow")) == pytest.approx(4.0, rel=0.03)
    rel = features.relative_powers(P, f)
    assert np.allclose(rel.sum(-1), 1.0, atol=0.02)


def test_reject_mask_drops_artifact_epoch():
    fs = 128.0
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, int(80 * fs)))
    x[1, 512 * 5: 512 * 6] *= 30
    ep = features.epoch(x, fs)
    f, P = features.epoch_psd(ep, fs)
    keep = features.reject_mask(P, f)
    assert not keep[5] and keep.sum() == len(keep) - 1


def test_git_blob_sha1():
    # `printf 'hello\n' | git hash-object --stdin`
    assert data.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_manifest_counts():
    rows = data.read_manifest()
    assert len(rows) == 121
    assert sum(r[0].startswith("ADHD") for r in rows) == 61


def test_icc1_recovers_variance_ratio():
    X, y, g = leakage.simulate_null(n_subjects=200, n_epochs=20, n_features=1,
                                    between_sd=1.0, within_sd=1.0, seed=5)
    assert stats.icc1(X[:, 0], g) == pytest.approx(0.5, abs=0.08)
