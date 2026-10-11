"""Does the result depend on the materiality threshold?

Relabels the same panel at several thresholds (no SEC rebuild needed) and
reruns the identical tune -> refit -> score-once procedure for each.

    python -m restatement_risk.sensitivity data/panel.parquet reports/sensitivity.json
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .panel import relabel
from .train import TOP_K, filter_panel, fit_and_score, split

THRESHOLDS = [0.0005, 0.001, 0.005, 0.01]  # share of total assets; 0.001 is the headline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("panel")
    p.add_argument("out")
    a = p.parse_args()
    raw = pd.read_parquet(a.panel)

    rows = []
    for t in THRESHOLDS:
        df = filter_panel(relabel(raw, t))
        train, val, test = split(df)
        _, _, _, res = fit_and_score(train, val, test)
        base = float(test["label"].mean())
        row = {"threshold": t, "test_rows": len(test), "test_positive_rate": round(base, 4)}
        for model in ("prior_revised_rule", "logistic", "gradient_boosting"):
            r = res[model]
            row[model] = {
                "pr_auc": round(r["pr_auc"], 4),
                "pr_auc_lift": round(r["pr_auc"] / base, 2),
                f"precision_at_{TOP_K}": round(r[f"precision_at_{TOP_K}"], 4),
            }
        rows.append(row)
        print(f"{t:.2%}  base {base:.3f}  logistic AP {row['logistic']['pr_auc']:.3f} "
              f"(x{row['logistic']['pr_auc_lift']})  GB AP {row['gradient_boosting']['pr_auc']:.3f} "
              f"(x{row['gradient_boosting']['pr_auc_lift']})  p@{TOP_K} {row['logistic'][f'precision_at_{TOP_K}']:.3f}",
              flush=True)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
