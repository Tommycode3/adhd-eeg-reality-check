"""Epoch-level vs subject-level cross-validation, and subject-identity decoding.

Everything runs single-threaded (n_jobs=1) on purpose.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_models(seed: int = 0) -> dict:
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed),
        "5-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def splits(scheme: str, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    if scheme == "epoch":
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y)
    if scheme == "subject":
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(y, y, groups)
    raise ValueError(scheme)


def cross_validate(model, X, y, groups, scheme: str, n_splits: int = 5, seed: int = 0) -> dict:
    """Return epoch-level balanced accuracy and subject-level (majority vote) accuracy.

    Subject-level accuracy is only meaningful for scheme='subject' (in
    epoch-CV a subject's epochs are spread over folds; we still aggregate the
    out-of-fold predictions per subject for comparability).
    """
    pred = np.empty_like(y)
    for tr, te in splits(scheme, y, groups, n_splits, seed):
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    subj = np.unique(groups)
    s_true = np.array([y[groups == s][0] for s in subj])
    s_pred = np.array([int(pred[groups == s].mean() > 0.5) for s in subj])
    return {
        "epoch_bacc": float(balanced_accuracy_score(y, pred)),
        "subject_bacc": float(balanced_accuracy_score(s_true, s_pred)),
    }


def subject_id_accuracy(model, X, groups, n_splits: int = 5, seed: int = 0) -> float:
    """Top-1 accuracy of predicting WHICH subject an epoch came from (epoch-level CV)."""
    pred = np.empty_like(groups)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, groups):
        pred[te] = clone(model).fit(X[tr], groups[tr]).predict(X[te])
    return float((pred == groups).mean())


def permute_subject_labels(y, groups, rng) -> np.ndarray:
    """Shuffle diagnosis labels between subjects (keeps every subject's epochs together)."""
    subj = np.unique(groups)
    lab = np.array([y[groups == s][0] for s in subj])
    perm = dict(zip(subj, rng.permutation(lab)))
    return np.array([perm[g] for g in groups])


def simulate_null(n_subjects: int = 40, n_epochs: int = 30, n_features: int = 10,
                  between_sd: float = 1.0, within_sd: float = 0.3, group_effect: float = 0.0,
                  seed: int = 0):
    """Synthetic EEG-like features: x_ij = mu + delta*y_i + b_i + e_ij.

    b_i ~ N(0, between_sd^2 I) is a subject 'fingerprint', e_ij ~ N(0, within_sd^2 I).
    With group_effect = 0 the label carries NO information about x.
    """
    rng = np.random.default_rng(seed)
    y_s = np.arange(n_subjects) % 2
    rng.shuffle(y_s)
    b = rng.normal(0, between_sd, (n_subjects, n_features))
    X, y, g = [], [], []
    for i in range(n_subjects):
        X.append(group_effect * y_s[i] + b[i] + rng.normal(0, within_sd, (n_epochs, n_features)))
        y += [y_s[i]] * n_epochs
        g += [i] * n_epochs
    return np.vstack(X), np.array(y), np.array(g)
