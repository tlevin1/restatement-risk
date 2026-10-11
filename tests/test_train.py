"""Evaluation helpers: the parts of train.py that decide what counts as a result."""

import math

import numpy as np
import pandas as pd

from restatement_risk.train import (
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    Winsorize,
    load,
    precision_at_k,
    split,
)


def test_splits_are_disjoint_and_in_time_order():
    assert max(TRAIN_YEARS) < min(VAL_YEARS)
    assert max(VAL_YEARS) < min(TEST_YEARS)


def test_split_assigns_each_year_once():
    df = pd.DataFrame({"fiscal_year": list(range(2011, 2024))})
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.fiscal_year).isdisjoint(test.fiscal_year)


def test_precision_at_k_averages_years_not_rows():
    # 2020: top-2 are both hits (1.0). 2021: top-2 has one hit (0.5). Mean 0.75,
    # regardless of how many rows each year has.
    df = pd.DataFrame({
        "fiscal_year": [2020] * 4 + [2021] * 3,
        "label":       [1, 1, 0, 0,   1, 0, 0],
    })
    scores = np.array([0.9, 0.8, 0.1, 0.2,   0.9, 0.8, 0.1])
    assert precision_at_k(df, scores, k=2) == 0.75


def test_winsorize_clips_to_training_percentiles_and_keeps_nans():
    train = np.array([[float(i)] for i in range(101)])  # 0..100
    w = Winsorize().fit(train)
    out = w.transform(np.array([[-50.0], [50.0], [500.0], [math.nan]]))
    assert out[0, 0] == 1.0 and out[1, 0] == 50.0 and out[2, 0] == 99.0
    assert math.isnan(out[3, 0])


def test_load_drops_unlabelled_small_spac_and_out_of_range_rows(tmp_path):
    big = math.log(50e6)
    rows = pd.DataFrame([
        {"fiscal_year": 2015, "label": 0.0, "log_assets": big, "spac": 0.0},             # kept
        {"fiscal_year": 2015, "label": math.nan, "log_assets": big, "spac": 0.0},        # unknown label
        {"fiscal_year": 2015, "label": 1.0, "log_assets": math.log(1e6), "spac": 0.0},   # under $10M
        {"fiscal_year": 2015, "label": 1.0, "log_assets": big, "spac": 1.0},             # SPAC
        {"fiscal_year": 2009, "label": 1.0, "log_assets": big, "spac": 0.0},             # before train years
    ])
    path = tmp_path / "panel.parquet"
    rows.to_parquet(path)
    assert len(load(str(path))) == 1
