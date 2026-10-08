"""Stream SEC's bulk companyfacts.zip into a company-year panel (parquet).

    python -m restatement_risk.build_panel data/raw/companyfacts.zip data/panel.parquet
"""

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .panel import DEFAULT_THRESHOLD, company_rows

_zip: zipfile.ZipFile | None = None


def _init(path: str) -> None:
    global _zip
    _zip = zipfile.ZipFile(path)


def _process(args: tuple[str, float]) -> list[dict]:
    name, threshold = args
    try:
        payload = json.loads(_zip.read(name))
        return company_rows(payload, threshold)
    except Exception:  # one malformed company file must not sink the build
        return []


def build(zip_path: str, threshold: float = DEFAULT_THRESHOLD, workers: int | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(zip_path,)) as ex:
        for i, out in enumerate(ex.map(_process, [(n, threshold) for n in names], chunksize=32)):
            rows.extend(out)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(names)} companies, {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("zip_path")
    p.add_argument("out")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    df = build(a.zip_path, a.threshold, a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"Wrote {len(df)} rows from {df['cik'].nunique()} companies to {a.out}")


if __name__ == "__main__":
    main()
