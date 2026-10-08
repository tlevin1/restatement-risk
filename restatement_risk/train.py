"""Train and evaluate restatement-risk models with a strict time split.

    python -m restatement_risk.train data/panel.parquet reports/

Hyperparameters are chosen on the validation years only; the final models are
refit on train+validation and scored once on the test years.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FEATURES = [
    "log_assets", "roa", "loss", "accruals", "equity_to_assets", "cash_to_assets",
    "soft_assets", "sales_growth", "d_receivables", "d_inventory", "asset_growth",
    "days_to_file", "n_concepts", "years_filing", "prior_revised",
    "prior_revision_size", "prior_amended", "prior_entity_change", "prior_spac",
    "bs_mismatch", "has_revenue", "has_inventory",
]
TRAIN_YEARS = range(2011, 2018)
VAL_YEARS = range(2018, 2020)
TEST_YEARS = range(2020, 2024)
MIN_ASSETS = 10e6  # drop shells and micro-entities whose filings are mostly noise
TOP_K = 100
SEED = 0


class Winsorize(BaseEstimator, TransformerMixin):
    """Clip each column to its training 1st/99th percentiles. Ratios like sales
    growth have extreme tails that would otherwise dominate a linear model."""

    def fit(self, X, y=None):
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def load(panel_path: str) -> pd.DataFrame:
    df = pd.read_parquet(panel_path)
    df = df[df["label"].notna()]
    df = df[np.exp(df["log_assets"]) >= MIN_ASSETS]
    # SPACs are shells, and their FY2020 wave came from an SEC reinterpretation of
    # warrant accounting (April 2021) that no fundamental feature could predict.
    df = df[df["spac"] == 0]
    return df[df["fiscal_year"].between(TRAIN_YEARS.start, TEST_YEARS.stop - 1)].reset_index(drop=True)


def split(df):
    pick = lambda years: df[df["fiscal_year"].isin(years)]  # noqa: E731
    return pick(TRAIN_YEARS), pick(VAL_YEARS), pick(TEST_YEARS)


def precision_at_k(df: pd.DataFrame, scores: np.ndarray, k: int = TOP_K) -> float:
    """Mean over years of the hit rate among the k highest-scored filings: what
    an analyst reviewing a fixed-size watchlist each year would see."""
    d = df.assign(score=scores)
    per_year = [g.nlargest(k, "score")["label"].mean() for _, g in d.groupby("fiscal_year")]
    return float(np.mean(per_year))


def bootstrap_ap(y, s, n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            out.append(average_precision_score(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def logistic(C):
    return make_pipeline(
        Winsorize(),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def boosting(lr, leaves, min_leaf):
    return HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        max_iter=500, early_stopping=True, validation_fraction=0.15,
        l2_regularization=1.0, random_state=SEED,
    )


@dataclass
class Choice:
    name: str
    params: dict
    val_ap: float


def tune(train, val) -> list[Choice]:
    Xtr, ytr, Xva, yva = train[FEATURES], train["label"], val[FEATURES], val["label"]
    choices = []
    best = max(
        ({"C": C} for C in (0.01, 0.1, 1.0, 10.0)),
        key=lambda p: average_precision_score(yva, logistic(**p).fit(Xtr, ytr).predict_proba(Xva)[:, 1]),
    )
    choices.append(Choice("logistic", best, average_precision_score(yva, logistic(**best).fit(Xtr, ytr).predict_proba(Xva)[:, 1])))
    grid = [{"lr": lr, "leaves": lv, "min_leaf": ml} for lr in (0.03, 0.1) for lv in (15, 31) for ml in (50, 200)]
    scored = [(average_precision_score(yva, boosting(**p).fit(Xtr, ytr).predict_proba(Xva)[:, 1]), p) for p in grid]
    ap, best = max(scored, key=lambda t: t[0])
    choices.append(Choice("gradient_boosting", best, ap))
    return choices


def evaluate(df_test, scores: dict[str, np.ndarray]) -> dict:
    y = df_test["label"].to_numpy()
    out = {}
    for name, s in scores.items():
        lo, hi = bootstrap_ap(y, s)
        out[name] = {
            "pr_auc": float(average_precision_score(y, s)),
            "pr_auc_95ci": [lo, hi],
            "roc_auc": float(roc_auc_score(y, s)),
            f"precision_at_{TOP_K}": precision_at_k(df_test, s),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("panel")
    p.add_argument("out_dir")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(a.panel)
    train, val, test = split(df)
    print({k: (len(v), round(v["label"].mean(), 4)) for k, v in {"train": train, "val": val, "test": test}.items()})

    choices = tune(train, val)
    for c in choices:
        print(f"  {c.name}: {c.params} val PR-AUC {c.val_ap:.4f}")

    fit_df = pd.concat([train, val])
    Xf, yf, Xt = fit_df[FEATURES], fit_df["label"], test[FEATURES]
    models = {
        "logistic": logistic(**choices[0].params).fit(Xf, yf),
        "gradient_boosting": boosting(**choices[1].params).fit(Xf, yf),
    }
    rng = np.random.default_rng(SEED)
    scores = {
        "base_rate": rng.random(len(test)),  # random ranking: PR-AUC ~ prevalence
        "prior_revised_rule": test["prior_revision_size"].fillna(-1).to_numpy() + test["prior_amended"].to_numpy(),
        **{n: m.predict_proba(Xt)[:, 1] for n, m in models.items()},
    }
    results = evaluate(test, scores)

    imp = permutation_importance(
        models["gradient_boosting"], Xt, test["label"], scoring="average_precision",
        n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    importance = sorted(
        ({"feature": f, "mean": float(m), "std": float(s)} for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std)),
        key=lambda d: -d["mean"],
    )

    by_year = (
        df.groupby("fiscal_year")
        .agg(rows=("label", "size"), positive_rate=("label", "mean"), revised_rate=("revised", "mean"), amended_rate=("amended", "mean"))
        .round(4).reset_index().to_dict(orient="records")
    )

    summary = {
        "universe": {"min_assets": MIN_ASSETS, "rows": len(df), "companies": int(df["cik"].nunique())},
        "split": {
            name: {"years": [r.start, r.stop - 1], "rows": len(d), "positive_rate": float(d["label"].mean())}
            for name, r, d in (("train", TRAIN_YEARS, train), ("val", VAL_YEARS, val), ("test", TEST_YEARS, test))
        },
        "tuning": [c.__dict__ for c in choices],
        "test": results,
        "permutation_importance_test": importance,
        "by_year": by_year,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in ("prior_revised_rule", "logistic", "gradient_boosting"):
        prec, rec, _ = precision_recall_curve(test["label"], scores[name])
        ax.plot(rec, prec, label=f"{name} (AP {results[name]['pr_auc']:.3f})")
    ax.axhline(test["label"].mean(), ls="--", c="grey", label=f"base rate ({test['label'].mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Test years {TEST_YEARS.start}-{TEST_YEARS.stop - 1}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=150)

    print(json.dumps(results, indent=2))
    print("Top features:", [d["feature"] for d in importance[:8]])


if __name__ == "__main__":
    main()
