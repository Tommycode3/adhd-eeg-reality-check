"""Dataset acquisition and loading.

Dataset: Nasrabadi, Allahverdy, Samavati & Mohammadi (2020),
"EEG data for ADHD / Control children", IEEE DataPort,
doi:10.21227/rzfh-zn36, license CC BY 4.0 (per the DataCite record).

The official copy needs a (free) IEEE DataPort login. Two routes are supported:

1. Official: download the four zips (ADHD_part1/2, Control_part1/2) from
   IEEE DataPort and put them in ``data/official/``. ``fetch()`` extracts them.
2. Mirror: ``fetch()`` downloads the same 121 ``.mat`` files from a public
   GitHub mirror pinned at a fixed commit, and verifies every file against
   the git blob SHA-1 in ``data/manifest.txt``. That manifest was cross-checked
   against a second, independent mirror: all 121 blob hashes are identical.

Raw data are never committed to this repository (see .gitignore).
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OFFICIAL_DIR = DATA_DIR / "official"
MANIFEST = DATA_DIR / "manifest.txt"

MIRROR_REPO = "MulindwaMarkMyles/machine-learning"
MIRROR_COMMIT = "296a1b39b1a05b902ff46df7c04de69a58c2bea2"
FS = 128.0
# Channel order documented on the IEEE DataPort page (Channel_Labels file).
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def read_manifest() -> list[tuple[str, str, int]]:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            rel, sha, size = line.split()
            rows.append((rel, sha, int(size)))
    return rows


def _extract_official() -> bool:
    zips = sorted(OFFICIAL_DIR.glob("*.zip")) if OFFICIAL_DIR.exists() else []
    if not zips:
        return False
    for z in zips:
        part = z.stem  # e.g. ADHD_part1
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.endswith(".mat"):
                    out = RAW_DIR / part / Path(name).name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))
    return True


def fetch(verbose: bool = True) -> None:
    """Populate data/raw/<part>/<file>.mat, verifying hashes."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if _extract_official():
        if verbose:
            print("Extracted official IEEE DataPort zips from data/official/")
    rows = read_manifest()
    mismatches = []
    for i, (rel, sha, size) in enumerate(rows, 1):
        out = RAW_DIR / rel
        if out.exists() and git_blob_sha1(out.read_bytes()) == sha:
            continue
        part = rel.split("/")[0]
        url = (f"https://raw.githubusercontent.com/{MIRROR_REPO}/{MIRROR_COMMIT}/"
               f"{part}/{rel}")
        if verbose:
            print(f"[{i}/{len(rows)}] {rel}")
        with urllib.request.urlopen(url, timeout=60) as r:
            content = r.read()
        if git_blob_sha1(content) != sha or len(content) != size:
            mismatches.append(rel)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
    if mismatches:
        raise RuntimeError(f"hash mismatch for {len(mismatches)} files: {mismatches[:5]}")
    if verbose:
        print(f"OK: {len(rows)} files verified in {RAW_DIR}")


@dataclass
class Recording:
    subject: str     # file stem, e.g. v10p
    group: int       # 1 = ADHD, 0 = control
    x: np.ndarray    # (n_channels, n_samples), microvolts as stored


def list_subjects() -> list[tuple[str, int, Path]]:
    out = []
    for rel, _, _ in read_manifest():
        p = RAW_DIR / rel
        group = 1 if rel.startswith("ADHD") else 0
        out.append((p.stem, group, p))
    return out


def load_recording(path: Path, group: int) -> Recording:
    m = loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one variable, got {keys}")
    x = np.asarray(m[keys[0]], dtype=np.float64)
    if x.shape[1] != len(CHANNELS):
        raise ValueError(f"{path}: unexpected shape {x.shape}")
    return Recording(subject=path.stem, group=group, x=x.T.copy())
