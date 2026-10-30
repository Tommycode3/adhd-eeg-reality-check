"""Check every DOI cited in the README against Crossref (DataCite for datasets).

Usage: python scripts/verify_dois.py   -> writes results/doi_check.tsv
Exit code 1 if any DOI does not resolve.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "adhd-eeg-audit/0.1 (mailto:maggiolotommaso@gmail.com)"}


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def lookup(doi: str) -> tuple[str, str, str]:
    q = urllib.parse.quote(doi)
    d = _get(f"https://api.crossref.org/works/{q}")
    if d:
        m = d["message"]
        title = (m.get("title") or [""])[0]
        cont = (m.get("container-title") or [""])[0]
        year = (m.get("issued", {}).get("date-parts") or [[""]])[0][0]
        return "crossref", f"{title} | {cont} | {year}", ""
    d = _get(f"https://api.datacite.org/dois/{q}")
    if d:
        a = d["data"]["attributes"]
        rights = "; ".join(r.get("rights", "") for r in a.get("rightsList", []))
        return "datacite", f"{a['titles'][0]['title']} | {a.get('publisher')} | {a.get('publicationYear')}", rights
    return "NOT FOUND", "", ""


def main() -> int:
    text = (ROOT / "README.md").read_text() + (ROOT / "CITATION.cff").read_text()
    dois = sorted(set(d.rstrip(").,") for d in re.findall(r"10\.\d{4,9}/[^\s)\]>\"']+", text)))
    out = ROOT / "results" / "doi_check.tsv"
    out.parent.mkdir(exist_ok=True)
    bad = 0
    lines = ["doi\tregistry\tmetadata\trights"]
    for doi in dois:
        reg, meta, rights = lookup(doi)
        bad += reg == "NOT FOUND"
        lines.append(f"{doi}\t{reg}\t{meta}\t{rights}")
        print(f"{reg:10s} {doi}  {meta[:110]}")
    out.write_text("\n".join(lines) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
