"""Synthetic companyfacts payloads: small enough to check every label by hand."""

import math

from restatement_risk.facts import CompanyFacts
from restatement_risk.panel import company_rows, revision_size


def row(val, end, accn, form="10-K", filed=None, start=None):
    r = {"end": end, "val": val, "accn": accn, "form": form, "filed": filed or f"{int(end[:4]) + 1}-02-15"}
    if start:
        r["start"] = start
    return r


def payload(concepts):
    return {"cik": 7, "entityName": "Test Co", "facts": {"us-gaap": {c: {"units": {"USD": rs}} for c, rs in concepts.items()}}}


A21, A22, A23 = "a-21", "a-22", "a-23"


def three_years(ni_2021_as_restated=100.0, amend=False):
    """FY2021-2023 10-Ks. Each also reports the prior year as a comparative,
    which is where a restatement of that prior year shows up."""
    ni = [
        row(100, "2021-12-31", A21, start="2021-01-01"),
        row(ni_2021_as_restated, "2021-12-31", A22, start="2021-01-01"),
        row(120, "2022-12-31", A22, start="2022-01-01"),
        row(120, "2022-12-31", A23, start="2022-01-01"),
        row(130, "2023-12-31", A23, start="2023-01-01"),
    ]
    assets = [
        row(10_000, "2021-12-31", A21),
        row(10_000, "2021-12-31", A22),
        row(11_000, "2022-12-31", A22),
        row(11_000, "2022-12-31", A23),
        row(12_000, "2023-12-31", A23),
    ]
    if amend:
        ni.append(row(80, "2021-12-31", "amend-21", form="10-K/A", filed="2022-06-01", start="2021-01-01"))
    return payload({"NetIncomeLoss": ni, "Assets": assets})


def by_year(rows):
    return {r["fiscal_year"]: r for r in rows}


def test_clean_history_labels_zero_and_last_year_unknown():
    rows = by_year(company_rows(three_years()))
    assert rows[2021]["label"] == 0 and rows[2022]["label"] == 0
    assert math.isnan(rows[2023]["label"])  # no 2024 10-K yet: unknown, not clean


def test_restated_comparative_is_labelled_revised():
    rows = by_year(company_rows(three_years(ni_2021_as_restated=70.0)))
    assert rows[2021]["revised"] == 1 and rows[2021]["label"] == 1
    assert rows[2021]["revision_size"] == 30 / 10_000


def test_revision_becomes_a_feature_only_for_the_next_year():
    rows = by_year(company_rows(three_years(ni_2021_as_restated=70.0)))
    assert math.isnan(rows[2021]["prior_revised"])  # nothing before 2021
    assert rows[2022]["prior_revised"] == 1  # revealed in the 2022 10-K itself
    assert rows[2022]["label"] == 0


def test_immaterial_revision_is_not_a_restatement():
    rows = by_year(company_rows(three_years(ni_2021_as_restated=99.5)))
    assert rows[2021]["revised"] == 0  # $0.5 on $10,000 assets is under 0.1%


def test_amendment_counts_even_without_a_next_filing():
    p = three_years(amend=True)
    rows = by_year(company_rows(p))
    assert rows[2021]["amended"] == 1 and rows[2021]["label"] == 1


def test_amendment_is_only_a_feature_once_filed():
    rows = by_year(company_rows(three_years(amend=True)))
    # The FY2021 10-K/A was filed 2022-06-01, before the FY2022 10-K (2023-02-15),
    # so it is public information when FY2022 is scored.
    assert rows[2022]["prior_amended"] == 1
    assert rows[2021]["prior_amended"] == 0


def test_original_filing_is_the_earliest_not_the_amendment():
    cf = CompanyFacts(three_years(amend=True))
    assert [f.accn for f in cf.annual_filings()] == [A21, A22, A23]


def test_revision_size_needs_assets():
    p = three_years()
    p["facts"]["us-gaap"].pop("Assets")
    cf = CompanyFacts(p)
    f = cf.annual_filings()
    assert revision_size(cf, f[0], f[1]) is None


def test_retailer_fiscal_year_convention():
    p = payload({"Assets": [row(1, "2024-02-03", "r-1")]})
    (r,) = company_rows(p)
    assert r["fiscal_year"] == 2023


def test_trust_account_marks_filing_as_spac():
    p = three_years()
    p["facts"]["us-gaap"]["AssetsHeldInTrustNoncurrent"] = {"units": {"USD": [row(9_000, "2021-12-31", A21)]}}
    rows = by_year(company_rows(p))
    assert rows[2021]["spac"] == 1 and rows[2022]["spac"] == 0


def test_reverse_merger_is_an_entity_change_not_a_restatement():
    # The FY2022 10-K's 2021 comparatives belong to the merger target: assets 10x larger.
    p = three_years(ni_2021_as_restated=-5_000.0)
    p["facts"]["us-gaap"]["Assets"]["units"]["USD"][1] = row(100_000, "2021-12-31", A22)
    rows = by_year(company_rows(p))
    assert math.isnan(rows[2021]["label"])  # unknowable, not a restatement
    assert rows[2022]["prior_entity_change"] == 1
    assert math.isnan(rows[2022]["prior_revision_size"])


def test_relabel_matches_building_at_that_threshold():
    import pandas as pd

    from restatement_risk.panel import relabel

    p = three_years(ni_2021_as_restated=99.0)  # a $1 revision on $10,000 of assets: 0.01%
    built_default = pd.DataFrame(company_rows(p))  # 0.1% threshold: not material
    for threshold in (0.00005, 0.001, 0.01):
        expected = pd.DataFrame(company_rows(p, threshold=threshold))
        pd.testing.assert_frame_equal(relabel(built_default, threshold), expected)
    assert relabel(built_default, 0.00005).loc[0, "label"] == 1  # material at 0.005%
