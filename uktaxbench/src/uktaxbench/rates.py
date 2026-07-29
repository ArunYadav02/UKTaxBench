"""
Rate tables for UK personal taxation.

EVERY figure in this file must be verified against a primary gov.uk source
before the benchmark is published. Each table carries a `source` URL and a
`verified` flag. `uktaxbench verify-rates` fails if anything is unverified.

Ground truth quality is the whole project. A benchmark built on wrong rates is
worse than no benchmark: it produces confident, citable, wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

Region = Literal["england_ni", "wales", "scotland"]
LoanPlan = Literal["plan1", "plan2", "plan4", "plan5", "pgl", "none"]

D = Decimal


@dataclass(frozen=True)
class Band:
    """A slice of income taxed at a single rate.

    `upper=None` means the band is unbounded above.
    Bounds are expressed in *taxable* income (i.e. after allowances), except
    where a table documents otherwise.
    """

    name: str
    lower: Decimal
    upper: Decimal | None
    rate: Decimal

    def width(self) -> Decimal | None:
        return None if self.upper is None else self.upper - self.lower


@dataclass(frozen=True)
class TaxYear:
    year: str
    personal_allowance: Decimal
    taper_threshold: Decimal
    taper_ratio: Decimal
    income_tax_bands: dict[str, list[Band]]
    ni_primary_threshold: Decimal
    ni_upper_earnings_limit: Decimal
    ni_main_rate: Decimal
    ni_upper_rate: Decimal
    dividend_allowance: Decimal
    dividend_rates: dict[str, Decimal]
    psa_basic: Decimal
    psa_higher: Decimal
    starting_rate_savings_band: Decimal
    hicbc_threshold: Decimal
    hicbc_upper: Decimal
    hicbc_step: Decimal
    child_benefit_first: Decimal
    child_benefit_additional: Decimal
    loan_thresholds: dict[str, Decimal]
    loan_rates: dict[str, Decimal]
    sources: dict[str, str] = field(default_factory=dict)
    verified: bool = False


# --------------------------------------------------------------------------
# 2025/26
# --------------------------------------------------------------------------
# STATUS: UNVERIFIED. These are the author's working figures. Before any
# results are published, check each against the linked gov.uk page and flip
# `verified=True`. See docs/RATE_VERIFICATION.md for the checklist.

TY_2025_26 = TaxYear(
    year="2025/26",
    personal_allowance=D("12570"),
    taper_threshold=D("100000"),
    taper_ratio=D("2"),  # £1 of allowance lost per £2 of income over threshold
    income_tax_bands={
        # Bounds are taxable income, i.e. measured after the personal allowance.
        "england_ni": [
            Band("basic", D("0"), D("37700"), D("0.20")),
            Band("higher", D("37700"), D("125140"), D("0.40")),
            Band("additional", D("125140"), None, D("0.45")),
        ],
        "wales": [
            Band("basic", D("0"), D("37700"), D("0.20")),
            Band("higher", D("37700"), D("125140"), D("0.40")),
            Band("additional", D("125140"), None, D("0.45")),
        ],
        # Scotland sets its own bands for non-savings, non-dividend income.
        # Published thresholds are gross; converted here to taxable-income
        # bounds by subtracting the standard personal allowance.
        "scotland": [
            Band("starter", D("0"), D("2827"), D("0.19")),
            Band("basic", D("2827"), D("14921"), D("0.20")),
            Band("intermediate", D("14921"), D("31092"), D("0.21")),
            Band("higher", D("31092"), D("62430"), D("0.42")),
            Band("advanced", D("62430"), D("112570"), D("0.45")),
            Band("top", D("112570"), None, D("0.48")),
        ],
    },
    ni_primary_threshold=D("12570"),
    ni_upper_earnings_limit=D("50270"),
    ni_main_rate=D("0.08"),
    ni_upper_rate=D("0.02"),
    dividend_allowance=D("500"),
    dividend_rates={
        "basic": D("0.0875"),
        "higher": D("0.3375"),
        "additional": D("0.3935"),
    },
    psa_basic=D("1000"),
    psa_higher=D("500"),
    starting_rate_savings_band=D("5000"),
    hicbc_threshold=D("60000"),
    hicbc_upper=D("80000"),
    hicbc_step=D("200"),  # 1% of child benefit per £200 over the threshold
    child_benefit_first=D("26.05"),  # per week
    child_benefit_additional=D("17.25"),  # per week, per additional child
    loan_thresholds={
        "plan1": D("26065"),
        "plan2": D("28470"),
        "plan4": D("32745"),
        "plan5": D("25000"),
        "pgl": D("21000"),
    },
    loan_rates={
        "plan1": D("0.09"),
        "plan2": D("0.09"),
        "plan4": D("0.09"),
        "plan5": D("0.09"),
        "pgl": D("0.06"),
    },
    sources={
        "income_tax": "https://www.gov.uk/income-tax-rates",
        "personal_allowance": "https://www.gov.uk/income-tax-rates/income-over-100000",
        "national_insurance": "https://www.gov.uk/national-insurance-rates-letters",
        "scotland": "https://www.gov.scot/publications/scottish-income-tax-2025-2026/",
        "student_loans": "https://www.gov.uk/repaying-your-student-loan/what-you-pay",
        "hicbc": "https://www.gov.uk/child-benefit-tax-charge",
        "child_benefit": "https://www.gov.uk/child-benefit/what-youll-get",
        "dividends": "https://www.gov.uk/tax-on-dividends",
        "savings": "https://www.gov.uk/apply-tax-free-interest-on-savings",
    },
    verified=False,
)


# --------------------------------------------------------------------------
# 2026/27  — the live tax year (6 April 2026 to 5 April 2027)
# --------------------------------------------------------------------------
# Income tax thresholds and NI are UNCHANGED from 2025/26: the personal
# allowance and higher-rate threshold are frozen until April 2031.
# Student loan thresholds DID rise for plans 1, 2 and 4. Plan 5 and PGL are
# unchanged. That asymmetry is the single most useful thing in this table:
# a model that has memorised "the 2026/27 answer" as "same as last year" will
# be right on income tax and wrong on student loans.
#
# Cross-checked against the House of Commons Library briefing on direct tax
# rates for 2026/27 and HMRC's Employer Bulletin (Dec 2025) figures for loan
# thresholds. Several commercial tax sites publish stale loan thresholds for
# this year; do not use them as a source. Flip `verified` only after checking
# the gov.uk pages listed in `sources`.

TY_2026_27 = TaxYear(
    year="2026/27",
    personal_allowance=D("12570"),
    taper_threshold=D("100000"),
    taper_ratio=D("2"),
    income_tax_bands={
        "england_ni": [
            Band("basic", D("0"), D("37700"), D("0.20")),
            Band("higher", D("37700"), D("125140"), D("0.40")),
            Band("additional", D("125140"), None, D("0.45")),
        ],
        "wales": [
            Band("basic", D("0"), D("37700"), D("0.20")),
            Band("higher", D("37700"), D("125140"), D("0.40")),
            Band("additional", D("125140"), None, D("0.45")),
        ],
        # Scottish starter/basic thresholds rose; intermediate and above are
        # frozen. Bounds below are taxable income (gross minus the standard
        # personal allowance). VERIFY against gov.scot before publishing.
        "scotland": [
            Band("starter", D("0"), D("2827"), D("0.19")),
            Band("basic", D("2827"), D("14921"), D("0.20")),
            Band("intermediate", D("14921"), D("31092"), D("0.21")),
            Band("higher", D("31092"), D("62430"), D("0.42")),
            Band("advanced", D("62430"), D("112570"), D("0.45")),
            Band("top", D("112570"), None, D("0.48")),
        ],
    },
    ni_primary_threshold=D("12570"),
    ni_upper_earnings_limit=D("50270"),
    ni_main_rate=D("0.08"),
    ni_upper_rate=D("0.02"),
    dividend_allowance=D("500"),
    dividend_rates={
        "basic": D("0.0875"),
        "higher": D("0.3375"),
        "additional": D("0.3935"),
    },
    psa_basic=D("1000"),
    psa_higher=D("500"),
    starting_rate_savings_band=D("5000"),
    hicbc_threshold=D("60000"),
    hicbc_upper=D("80000"),
    hicbc_step=D("200"),
    child_benefit_first=D("26.05"),
    child_benefit_additional=D("17.25"),
    loan_thresholds={
        "plan1": D("26900"),   # up from 26065
        "plan2": D("29385"),   # up from 28470
        "plan4": D("33795"),   # up from 32745
        "plan5": D("25000"),   # frozen
        "pgl": D("21000"),     # never changed since introduction
    },
    loan_rates={
        "plan1": D("0.09"),
        "plan2": D("0.09"),
        "plan4": D("0.09"),
        "plan5": D("0.09"),
        "pgl": D("0.06"),
    },
    sources={
        "income_tax": "https://www.gov.uk/income-tax-rates",
        "personal_allowance": "https://www.gov.uk/income-tax-rates/income-over-100000",
        "national_insurance": "https://www.gov.uk/national-insurance-rates-letters",
        "scotland": "https://www.gov.scot/publications/scottish-income-tax-2026-2027/",
        "student_loans": "https://www.gov.uk/repaying-your-student-loan/what-you-pay",
        "hicbc": "https://www.gov.uk/child-benefit-tax-charge",
        "child_benefit": "https://www.gov.uk/child-benefit/what-youll-get",
        "dividends": "https://www.gov.uk/tax-on-dividends",
        "savings": "https://www.gov.uk/apply-tax-free-interest-on-savings",
    },
    verified=False,
)


YEARS: dict[str, TaxYear] = {
    "2025/26": TY_2025_26,
    "2026/27": TY_2026_27,
}

CURRENT_YEAR = "2026/27"
PRIOR_YEAR = "2025/26"


def figures_that_changed(a: str, b: str) -> dict[str, tuple[Decimal, Decimal]]:
    """Fields that differ between two tax years.

    Used by the grader for stale-year attribution: if a model's answer matches
    the prior year's ground truth but not the current year's, we can say which
    specific threshold it used rather than just marking it wrong.
    """
    ya, yb = get_year(a), get_year(b)
    diffs: dict[str, tuple[Decimal, Decimal]] = {}
    for fld in (
        "personal_allowance",
        "taper_threshold",
        "ni_primary_threshold",
        "ni_upper_earnings_limit",
        "ni_main_rate",
        "ni_upper_rate",
        "dividend_allowance",
        "hicbc_threshold",
        "hicbc_upper",
        "child_benefit_first",
    ):
        va, vb = getattr(ya, fld), getattr(yb, fld)
        if va != vb:
            diffs[fld] = (va, vb)
    for plan, va in ya.loan_thresholds.items():
        vb = yb.loan_thresholds.get(plan)
        if vb is not None and va != vb:
            diffs[f"loan_threshold_{plan}"] = (va, vb)
    return diffs


def get_year(year: str = "2025/26") -> TaxYear:
    if year not in YEARS:
        raise KeyError(
            f"No rate table for {year!r}. Available: {sorted(YEARS)}. "
            "Add one in rates.py rather than guessing at runtime."
        )
    return YEARS[year]


def unverified_years() -> list[str]:
    return [y for y, t in YEARS.items() if not t.verified]
