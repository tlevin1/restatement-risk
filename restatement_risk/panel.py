"""Turn one company's facts into company-year rows: features known on the
original filing date, plus the label revealed afterwards.

Leakage rule: a feature for fiscal year t may only use the original 10-K for t
(which includes year t-1 comparatives) and anything filed before it. The label
uses the next 10-K and any 10-K/A, which are filed later.
"""

from __future__ import annotations

import math
from datetime import date

from .facts import CompanyFacts, Filing, next_annual, prior_annual

NET_INCOME = ["NetIncomeLoss", "ProfitLoss"]
EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
ASSETS = ["Assets"]
LIABILITIES_AND_EQUITY = ["LiabilitiesAndStockholdersEquity"]
REVENUE = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
CFO = ["NetCashProvidedByUsedInOperatingActivities"]
RECEIVABLES = ["AccountsReceivableNetCurrent"]
INVENTORY = ["InventoryNet"]
PPE = ["PropertyPlantAndEquipmentNet"]
CASH = ["CashAndCashEquivalentsAtCarryingValue"]

# A revision counts as material when net income or equity moves by more than
# this fraction of originally reported total assets. Scaling by assets rather
# than by the item itself keeps near-zero net income from turning rounding
# into "restatements".
DEFAULT_THRESHOLD = 0.001


# If total assets for the same date differ by more than this factor between two
# filings, the later filing is describing a different entity (a reverse merger
# or de-SPAC made the target the accounting predecessor), not revising numbers.
ENTITY_CHANGE_FACTOR = 2.0


def entity_changed(cf: CompanyFacts, orig: Filing, later: Filing) -> bool:
    a = cf.instant(ASSETS, orig.accn, orig.period_end)
    b = cf.instant(ASSETS, later.accn, orig.period_end)
    if not a or not b or a <= 0 or b <= 0:
        return False
    return max(a / b, b / a) > ENTITY_CHANGE_FACTOR


def revision_size(cf: CompanyFacts, orig: Filing, later: Filing) -> float | None:
    """Largest change in net income or equity for orig's period, as reported in
    `later` versus `orig`, scaled by originally reported assets. None when it
    cannot be measured, including when the later filing is a different entity."""
    assets = cf.instant(ASSETS, orig.accn, orig.period_end)
    if not assets or assets <= 0 or entity_changed(cf, orig, later):
        return None
    deltas = []
    for get, concepts in ((cf.annual, NET_INCOME), (cf.instant, EQUITY)):
        a = get(concepts, orig.accn, orig.period_end)
        b = get(concepts, later.accn, orig.period_end)
        if a is not None and b is not None:
            deltas.append(abs(b - a) / assets)
    return max(deltas) if deltas else None


def _ratio(num, den):
    if num is None or den is None or den == 0:
        return math.nan
    return num / den


def _nan(x):
    return math.nan if x is None else x


def features(cf: CompanyFacts, filings: list[Filing], i: int, threshold: float) -> dict:
    f = filings[i]
    acc, end = f.accn, f.period_end
    prior = prior_annual(filings, i)
    prev_end = prior.period_end if prior else None

    assets = cf.instant(ASSETS, acc, end)
    ni = cf.annual(NET_INCOME, acc, end)
    eq = cf.instant(EQUITY, acc, end)
    cfo = cf.annual(CFO, acc, end)
    rev = cf.annual(REVENUE, acc, end)
    cash = cf.instant(CASH, acc, end)
    ppe = cf.instant(PPE, acc, end)
    le = cf.instant(LIABILITIES_AND_EQUITY, acc, end)

    # Prior-year comparatives come from this same filing, so they are known now.
    def prev(get, concepts):
        return get(concepts, acc, prev_end) if prev_end else None

    rev_prev = prev(cf.annual, REVENUE)
    ar, ar_prev = cf.instant(RECEIVABLES, acc, end), prev(cf.instant, RECEIVABLES)
    inv, inv_prev = cf.instant(INVENTORY, acc, end), prev(cf.instant, INVENTORY)
    assets_prev = prev(cf.instant, ASSETS)

    prior_rev = revision_size(cf, prior, f) if prior else None
    prior_entity_change = bool(prior and entity_changed(cf, prior, f))
    prior_amended = bool(prior and any(d < f.filed for d in cf.amendments_for(prior.period_end)))

    soft = None
    if assets and cash is not None:
        soft = (assets - cash - (ppe or 0)) / assets

    return {
        "log_assets": math.log(assets) if assets and assets > 0 else math.nan,
        "roa": _ratio(ni, assets),
        "loss": math.nan if ni is None else float(ni < 0),
        "accruals": _ratio(None if ni is None or cfo is None else ni - cfo, assets),
        "equity_to_assets": _ratio(eq, assets),
        "cash_to_assets": _ratio(cash, assets),
        "soft_assets": _nan(soft),
        "sales_growth": _ratio(None if rev is None or rev_prev is None else rev - rev_prev, rev_prev),
        "d_receivables": _ratio(None if ar is None or ar_prev is None else ar - ar_prev, assets),
        "d_inventory": _ratio(None if inv is None or inv_prev is None else inv - inv_prev, assets),
        "asset_growth": _ratio(None if assets is None or assets_prev is None else assets - assets_prev, assets_prev),
        "days_to_file": float((f.filed - end).days),
        "n_concepts": float(cf.concepts_per_accn.get(acc, 0)),
        "years_filing": float(i),
        "prior_revised": math.nan if prior_rev is None else float(prior_rev > threshold),
        "prior_revision_size": _nan(prior_rev),
        "prior_amended": float(prior_amended),
        "prior_entity_change": float(prior_entity_change),
        "prior_spac": float(bool(prior and prior.accn in cf.trust_accns)),
        "bs_mismatch": math.nan if assets is None or le is None else float(abs(assets - le) > max(1e6, 1e-4 * assets)),
        "has_revenue": float(rev is not None),
        "has_inventory": float(inv is not None),
    }


def label(cf: CompanyFacts, filings: list[Filing], i: int, threshold: float) -> dict:
    f = filings[i]
    nxt = next_annual(filings, i)
    size = revision_size(cf, f, nxt) if nxt else None
    amended = any(d > f.filed for d in cf.amendments_for(f.period_end))
    revised = size is not None and size > threshold
    # An entity change makes the comparison meaningless, so unless the year was
    # amended outright the label is unknown rather than "clean".
    known = amended or (size is not None)
    return {
        "revision_size": _nan(size),
        "revised": float(revised),
        "amended": float(amended),
        # Unknown when there is no next 10-K to compare against and no amendment:
        # those rows are dropped, never assumed clean.
        "label": float(revised or amended) if known else math.nan,
    }


def company_rows(payload: dict, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    cf = CompanyFacts(payload)
    filings = cf.annual_filings()
    rows = []
    for i, f in enumerate(filings):
        if cf.instant(ASSETS, f.accn, f.period_end) is None:
            continue  # no balance sheet tagged: nothing to model
        rows.append({
            "cik": cf.cik,
            "name": cf.name,
            "accn": f.accn,
            "period_end": f.period_end.isoformat(),
            "filed": f.filed.isoformat(),
            "fiscal_year": f.fiscal_year,
            "spac": float(f.accn in cf.trust_accns),
            **features(cf, filings, i, threshold),
            **label(cf, filings, i, threshold),
        })
    return rows
