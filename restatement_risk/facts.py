"""Indexed access to one company's EDGAR companyfacts payload.

Every lookup is scoped to a single filing (accession number), so a value is
always "as reported in that document". That is what lets us compare a number
as originally filed with the same number as reported a year later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 350, 380
NEXT_FILING_MIN_DAYS, NEXT_FILING_MAX_DAYS = 330, 400


@dataclass(frozen=True)
class Filing:
    accn: str
    form: str
    filed: date
    period_end: date

    @property
    def fiscal_year(self) -> int:
        # Compustat convention: years ending January-May belong to the prior year,
        # so a retailer's year ending 2024-02-01 is fiscal 2023.
        return self.period_end.year if self.period_end.month > 5 else self.period_end.year - 1


def _d(s: str) -> date:
    return date.fromisoformat(s)


class CompanyFacts:
    def __init__(self, payload: dict):
        self.cik = int(payload.get("cik", 0))
        self.name = payload.get("entityName", "")
        # concept -> accn -> rows (USD only)
        self._idx: dict[str, dict[str, list[dict]]] = {}
        # accn -> number of distinct concepts reported (a crude complexity measure)
        self.concepts_per_accn: dict[str, int] = defaultdict(int)
        self._filings: dict[str, Filing] = {}
        self._amendment_dates: dict[date, list[date]] = defaultdict(list)
        # Filings with a trust account on the balance sheet: blank-check shells (SPACs)
        self.trust_accns: set[str] = set()

        for concept, node in payload.get("facts", {}).get("us-gaap", {}).items():
            by_accn: dict[str, list[dict]] = defaultdict(list)
            for unit, rows in node.get("units", {}).items():
                for r in rows:
                    form = r.get("form", "")
                    if unit == "USD":
                        by_accn[r["accn"]].append(r)
                    if form in ("10-K", "10-K/A"):
                        self._note_filing(r)
            for accn in by_accn:
                self.concepts_per_accn[accn] += 1
            if concept.startswith("AssetsHeldInTrust"):
                self.trust_accns.update(by_accn)
            self._idx[concept] = dict(by_accn)

        for f in self._filings.values():
            if f.form == "10-K/A":
                self._amendment_dates[f.period_end].append(f.filed)

    def _note_filing(self, r: dict) -> None:
        end = _d(r["end"])
        prev = self._filings.get(r["accn"])
        if prev is None or end > prev.period_end:
            self._filings[r["accn"]] = Filing(r["accn"], r["form"], _d(r["filed"]), end)

    def annual_filings(self) -> list[Filing]:
        """Original 10-Ks, oldest first, one per period end (the earliest filed)."""
        by_end: dict[date, Filing] = {}
        for f in self._filings.values():
            if f.form != "10-K":
                continue
            if f.period_end not in by_end or f.filed < by_end[f.period_end].filed:
                by_end[f.period_end] = f
        return sorted(by_end.values(), key=lambda f: f.period_end)

    def amendments_for(self, period_end: date) -> list[date]:
        """Filing dates of 10-K/As that carry financial data for this period."""
        return sorted(self._amendment_dates.get(period_end, []))

    def _rows(self, concept: str, accn: str) -> list[dict]:
        return self._idx.get(concept, {}).get(accn, [])

    def instant(self, concepts: list[str], accn: str, end: date) -> float | None:
        end_s = end.isoformat()
        for c in concepts:
            for r in self._rows(c, accn):
                if r["end"] == end_s and "start" not in r:
                    return float(r["val"])
        return None

    def annual(self, concepts: list[str], accn: str, end: date) -> float | None:
        """A roughly one-year duration value ending at `end`."""
        end_s = end.isoformat()
        for c in concepts:
            for r in self._rows(c, accn):
                if r["end"] != end_s or "start" not in r:
                    continue
                days = (end - _d(r["start"])).days + 1
                if ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS:
                    return float(r["val"])
        return None


def next_annual(filings: list[Filing], i: int) -> Filing | None:
    """The 10-K covering the year after filings[i], if the company filed one."""
    cur = filings[i]
    for f in filings[i + 1 :]:
        gap = (f.period_end - cur.period_end).days
        if NEXT_FILING_MIN_DAYS <= gap <= NEXT_FILING_MAX_DAYS:
            return f
        if gap > NEXT_FILING_MAX_DAYS:
            return None
    return None


def prior_annual(filings: list[Filing], i: int) -> Filing | None:
    cur = filings[i]
    for f in reversed(filings[:i]):
        gap = (cur.period_end - f.period_end).days
        if NEXT_FILING_MIN_DAYS <= gap <= NEXT_FILING_MAX_DAYS:
            return f
        if gap > NEXT_FILING_MAX_DAYS:
            return None
    return None
