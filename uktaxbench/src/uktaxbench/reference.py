"""
Reference implementation: the benchmark's ground truth.

This is a *benchmark artifact*, not tax advice and not a payroll engine. It
models the common PAYE case on an annual basis. Documented simplifications:

  1. NI is computed annually. Real Class 1 NI is computed per pay period and
     is not cumulative, so a worker with uneven pay pays a different amount
     than this function returns. Cases involving irregular pay are excluded
     from the generated set for exactly this reason.
  2. Only Class 1 employee NI. No Class 1A/1B, no employer NI, no directors'
     annualised rules, no married couple's allowance, no blind person's
     allowance, no gift aid, no EIS/VCT/SEIS relief.
  3. Student loan repayments are computed on annual income and rounded down
     to whole pounds, matching HMRC's stated method.
  4. Scottish rates apply to non-savings, non-dividend income only. Savings
     and dividends use UK-wide rates and UK-wide band widths for deciding
     which rate applies — this is genuinely how it works, and it is the
     single most common thing people get wrong about Scottish tax.

Money is Decimal throughout. Floats will produce off-by-a-penny errors that
propagate into ground truth and quietly corrupt the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any

from .rates import Band, LoanPlan, Region, TaxYear, get_year

D = Decimal
ZERO = D("0")
PENNY = D("0.01")


def _p(x: Decimal) -> Decimal:
    """Round to the penny, half up."""
    return x.quantize(PENNY, rounding=ROUND_HALF_UP)


def _pound_down(x: Decimal) -> Decimal:
    """Round down to whole pounds — HMRC's method for student loans."""
    return x.quantize(D("1"), rounding=ROUND_DOWN)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Taxpayer:
    """One taxpayer-year.

    All amounts are annual and in pounds.

    salary                  gross employment income before any sacrifice
    salary_sacrifice        given up under a valid arrangement; reduces gross
                            pay for BOTH income tax and NI
    pension_relief_at_source
                            personal pension contribution paid from net pay.
                            Grossed up by 25% for band extension and for
                            adjusted net income. Does NOT reduce NI.
    pension_net_pay         occupational scheme deducted before tax but after
                            NI. Reduces taxable pay, NOT NI-able pay.
    savings_interest        gross bank/building society interest
    dividends               gross UK dividend income
    other_income            non-savings non-dividend income outside PAYE
                            (property, self-employment profit). Not NI-able
                            here; Class 2/4 NI is out of scope.
    loan_plans              student loan plans being repaid simultaneously
    children                number of children Child Benefit is claimed for
    claims_child_benefit    whether the household actually claims it
    region                  determines which income tax band table applies
    """

    salary: Decimal = ZERO
    salary_sacrifice: Decimal = ZERO
    pension_relief_at_source: Decimal = ZERO
    pension_net_pay: Decimal = ZERO
    savings_interest: Decimal = ZERO
    dividends: Decimal = ZERO
    other_income: Decimal = ZERO
    loan_plans: tuple[LoanPlan, ...] = ()
    children: int = 0
    claims_child_benefit: bool = False
    region: Region = "england_ni"
    year: str = "2025/26"

    def __post_init__(self) -> None:
        if self.salary_sacrifice > self.salary:
            raise ValueError("salary_sacrifice cannot exceed salary")
        if self.children < 0:
            raise ValueError("children cannot be negative")
        if "none" in self.loan_plans and len(self.loan_plans) > 1:
            raise ValueError("'none' cannot be combined with other plans")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


@dataclass
class TaxResult:
    gross_pay_after_sacrifice: Decimal = ZERO
    total_income: Decimal = ZERO
    adjusted_net_income: Decimal = ZERO
    personal_allowance: Decimal = ZERO
    allowance_lost_to_taper: Decimal = ZERO
    taxable_income: Decimal = ZERO

    income_tax: Decimal = ZERO
    tax_on_earnings: Decimal = ZERO
    tax_on_savings: Decimal = ZERO
    tax_on_dividends: Decimal = ZERO

    national_insurance: Decimal = ZERO
    student_loan: Decimal = ZERO
    student_loan_by_plan: dict[str, Decimal] = field(default_factory=dict)

    child_benefit_received: Decimal = ZERO
    hicbc: Decimal = ZERO

    total_deductions: Decimal = ZERO
    take_home: Decimal = ZERO

    marginal_rate: Decimal = ZERO
    effective_rate: Decimal = ZERO

    band_breakdown: list[dict[str, Any]] = field(default_factory=list)
    rules_engaged: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, Decimal):
                out[k] = float(v)
            elif isinstance(v, dict):
                out[k] = {
                    kk: float(vv) if isinstance(vv, Decimal) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                out[k] = [
                    {
                        kk: float(vv) if isinstance(vv, Decimal) else vv
                        for kk, vv in item.items()
                    }
                    if isinstance(item, dict)
                    else item
                    for item in v
                ]
            else:
                out[k] = v
        return out


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def adjusted_net_income(tp: Taxpayer, ty: TaxYear | None = None) -> Decimal:
    """Income after grossed-up relief-at-source pension contributions.

    This figure — not gross salary — drives both the personal allowance taper
    and the High Income Child Benefit Charge. Conflating the two is one of the
    most common real-world errors, so the benchmark tests it directly.
    """
    ty = ty or get_year(tp.year)
    gross_pension_rc = tp.pension_relief_at_source * D("1.25")
    total = (
        (tp.salary - tp.salary_sacrifice)
        - tp.pension_net_pay
        + tp.savings_interest
        + tp.dividends
        + tp.other_income
    )
    return max(ZERO, total - gross_pension_rc)


def personal_allowance(ani: Decimal, ty: TaxYear) -> tuple[Decimal, Decimal]:
    """Return (allowance, amount lost to taper)."""
    if ani <= ty.taper_threshold:
        return ty.personal_allowance, ZERO
    excess = ani - ty.taper_threshold
    lost = min(ty.personal_allowance, (excess / ty.taper_ratio))
    lost = lost.quantize(PENNY, rounding=ROUND_HALF_UP)
    return ty.personal_allowance - lost, lost


def _tax_bands(
    amount: Decimal, bands: list[Band], already_used: Decimal
) -> tuple[Decimal, list[dict[str, Any]], Decimal]:
    """Tax `amount` sitting on top of `already_used` of taxable income.

    Returns (tax, breakdown rows, new cumulative used).
    """
    tax = ZERO
    rows: list[dict[str, Any]] = []
    remaining = amount
    cursor = already_used

    for band in bands:
        if remaining <= ZERO:
            break
        band_top = band.upper if band.upper is not None else cursor + remaining
        if cursor >= band_top:
            continue
        slice_amount = min(remaining, band_top - cursor)
        if slice_amount <= ZERO:
            continue
        band_tax = _p(slice_amount * band.rate)
        tax += band_tax
        rows.append(
            {
                "band": band.name,
                "rate": band.rate,
                "amount": slice_amount,
                "tax": band_tax,
            }
        )
        cursor += slice_amount
        remaining -= slice_amount

    return tax, rows, cursor


def _savings_allowance(total_taxable: Decimal, ty: TaxYear) -> Decimal:
    """Personal savings allowance, set by which UK band total taxable income
    reaches: £1,000 basic, £500 higher, nil additional."""
    uk_bands = ty.income_tax_bands["england_ni"]
    basic_top = uk_bands[0].upper or ZERO
    higher_top = uk_bands[1].upper or ZERO
    if total_taxable > higher_top:
        return ZERO
    if total_taxable > basic_top:
        return ty.psa_higher
    return ty.psa_basic


def _savings_zero_rate(
    taxable_earnings: Decimal,
    taxable_savings_pre: Decimal,
    total_taxable: Decimal,
    ty: TaxYear,
) -> tuple[Decimal, Decimal]:
    """Split savings income into (zero-rated, taxable).

    Two separate reliefs stack here and models routinely collapse them into
    one. The starting rate for savings gives a 0% band that shrinks pound for
    pound as non-savings income rises above the personal allowance; the
    personal savings allowance then applies on top of whatever is left.
    """
    srs_available = max(ZERO, ty.starting_rate_savings_band - taxable_earnings)
    srs_used = min(taxable_savings_pre, srs_available)
    after_srs = taxable_savings_pre - srs_used
    psa = _savings_allowance(total_taxable, ty)
    psa_used = min(after_srs, psa)
    return srs_used + psa_used, after_srs - psa_used


def national_insurance(tp: Taxpayer, ty: TaxYear | None = None) -> Decimal:
    """Class 1 employee NI, annualised.

    NI-able pay is salary after salary sacrifice. Net-pay-arrangement pension
    contributions and relief-at-source contributions do NOT reduce it — this
    asymmetry is a deliberate difficulty axis in the benchmark.
    """
    ty = ty or get_year(tp.year)
    niable = max(ZERO, tp.salary - tp.salary_sacrifice)
    if niable <= ty.ni_primary_threshold:
        return ZERO
    main_slice = min(niable, ty.ni_upper_earnings_limit) - ty.ni_primary_threshold
    upper_slice = max(ZERO, niable - ty.ni_upper_earnings_limit)
    return _p(main_slice * ty.ni_main_rate + upper_slice * ty.ni_upper_rate)


def student_loan(
    tp: Taxpayer, ty: TaxYear | None = None
) -> tuple[Decimal, dict[str, Decimal]]:
    """Repayments across all plans held simultaneously.

    A borrower with both an undergraduate plan and a postgraduate loan repays
    on both at once, each against its own threshold. Models frequently return
    only the larger of the two.

    Repayment income here is pay after salary sacrifice plus unearned income;
    relief-at-source pension contributions do not reduce it.
    """
    ty = ty or get_year(tp.year)
    plans = [p for p in tp.loan_plans if p != "none"]
    if not plans:
        return ZERO, {}

    income = (
        max(ZERO, tp.salary - tp.salary_sacrifice)
        - tp.pension_net_pay
        + tp.savings_interest
        + tp.dividends
        + tp.other_income
    )

    per_plan: dict[str, Decimal] = {}
    total = ZERO
    for plan in plans:
        threshold = ty.loan_thresholds[plan]
        rate = ty.loan_rates[plan]
        amount = _pound_down(max(ZERO, income - threshold) * rate)
        per_plan[plan] = amount
        total += amount
    return total, per_plan


def child_benefit_amount(children: int, ty: TaxYear) -> Decimal:
    if children <= 0:
        return ZERO
    weekly = ty.child_benefit_first + ty.child_benefit_additional * (children - 1)
    return _p(weekly * D("52"))


def hicbc(tp: Taxpayer, ani: Decimal, ty: TaxYear | None = None) -> tuple[Decimal, Decimal]:
    """High Income Child Benefit Charge. Returns (benefit received, charge).

    The charge is 1% of the benefit for every £200 of adjusted net income
    above the threshold, with the *number of whole £200 steps* counted — not a
    smooth proportion. Rounding here is a real difficulty axis.
    """
    ty = ty or get_year(tp.year)
    benefit = child_benefit_amount(tp.children, ty)
    if not tp.claims_child_benefit or benefit == ZERO:
        return ZERO if not tp.claims_child_benefit else benefit, ZERO
    if ani <= ty.hicbc_threshold:
        return benefit, ZERO
    if ani >= ty.hicbc_upper:
        return benefit, _p(benefit)
    steps = int((ani - ty.hicbc_threshold) / ty.hicbc_step)
    pct = D(steps) / D("100")
    return benefit, _p(benefit * pct)


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


def compute(tp: Taxpayer) -> TaxResult:
    ty = get_year(tp.year)
    r = TaxResult()
    engaged: list[str] = []

    gross = max(ZERO, tp.salary - tp.salary_sacrifice)
    r.gross_pay_after_sacrifice = gross
    if tp.salary_sacrifice > ZERO:
        engaged.append("salary_sacrifice")
    if tp.pension_relief_at_source > ZERO:
        engaged.append("pension_relief_at_source")
    if tp.pension_net_pay > ZERO:
        engaged.append("pension_net_pay")

    earnings = gross - tp.pension_net_pay + tp.other_income
    r.total_income = earnings + tp.savings_interest + tp.dividends

    ani = adjusted_net_income(tp, ty)
    r.adjusted_net_income = ani

    pa, lost = personal_allowance(ani, ty)
    r.personal_allowance = pa
    r.allowance_lost_to_taper = lost
    if lost > ZERO:
        engaged.append("personal_allowance_taper")

    # Allowance is set against non-savings income first, then savings, then
    # dividends — the order that produces the lowest bill in the ordinary case.
    remaining_pa = pa
    taxable_earnings = max(ZERO, earnings - remaining_pa)
    remaining_pa = max(ZERO, remaining_pa - earnings)
    taxable_savings_pre = max(ZERO, tp.savings_interest - remaining_pa)
    remaining_pa = max(ZERO, remaining_pa - tp.savings_interest)
    taxable_dividends_pre = max(ZERO, tp.dividends - remaining_pa)

    r.taxable_income = taxable_earnings + taxable_savings_pre + taxable_dividends_pre

    # Relief-at-source contributions extend the basic rate band.
    band_extension = tp.pension_relief_at_source * D("1.25")
    bands = list(ty.income_tax_bands[tp.region])
    if band_extension > ZERO:
        bands = _extend_bands(bands, band_extension)

    # 1. Non-savings, non-dividend income (regional rates).
    tax_earn, rows_earn, used = _tax_bands(taxable_earnings, bands, ZERO)
    r.tax_on_earnings = tax_earn

    # 2. Savings, stacked on top. UK-wide rates even in Scotland.
    uk_bands = list(ty.income_tax_bands["england_ni"])
    if band_extension > ZERO:
        uk_bands = _extend_bands(uk_bands, band_extension)
    if tp.savings_interest > ZERO:
        engaged.append("savings_interest")
    zero_rated_savings, taxable_savings = _savings_zero_rate(
        taxable_earnings, taxable_savings_pre, r.taxable_income, ty
    )
    if zero_rated_savings > ZERO and taxable_earnings < ty.starting_rate_savings_band:
        engaged.append("starting_rate_for_savings")
    tax_sav, rows_sav, used = _tax_bands(
        taxable_savings, uk_bands, used + zero_rated_savings
    )
    r.tax_on_savings = tax_sav

    # 3. Dividends on top of everything. UK-wide rates even in Scotland.
    if tp.dividends > ZERO:
        engaged.append("dividends")
    div_allow = min(ty.dividend_allowance, taxable_dividends_pre)
    taxable_dividends = taxable_dividends_pre - div_allow
    div_bands = _dividend_bands(ty, band_extension)
    tax_div, rows_div, used = _tax_bands(
        taxable_dividends, div_bands, used + div_allow
    )
    r.tax_on_dividends = tax_div

    r.income_tax = _p(tax_earn + tax_sav + tax_div)
    r.band_breakdown = rows_earn + rows_sav + rows_div

    r.national_insurance = national_insurance(tp, ty)
    r.student_loan, r.student_loan_by_plan = student_loan(tp, ty)
    if r.student_loan_by_plan:
        engaged.append("student_loan")
        if len(r.student_loan_by_plan) > 1:
            engaged.append("multiple_loan_plans")

    benefit, charge = hicbc(tp, ani, ty)
    r.child_benefit_received = benefit
    r.hicbc = charge
    if charge > ZERO:
        engaged.append("hicbc")

    if tp.region == "scotland":
        engaged.append("scottish_rates")

    r.total_deductions = _p(
        r.income_tax + r.national_insurance + r.student_loan + r.hicbc
    )
    r.take_home = _p(
        gross
        - tp.pension_net_pay
        + tp.other_income
        + tp.savings_interest
        + tp.dividends
        - r.total_deductions
    )

    r.marginal_rate = _marginal_rate(tp)
    if r.total_income > ZERO:
        r.effective_rate = (r.total_deductions / r.total_income).quantize(
            D("0.0001"), rounding=ROUND_HALF_UP
        )
    r.rules_engaged = engaged
    return r


def _extend_bands(bands: list[Band], extension: Decimal) -> list[Band]:
    """Push every threshold except the additional-rate floor up by `extension`.

    Relief-at-source pension contributions widen the basic rate band. The
    additional rate threshold is not extended in the same way for the purposes
    of this model.
    """
    out: list[Band] = []
    cursor = ZERO
    for i, b in enumerate(bands):
        upper = None if b.upper is None else b.upper + extension
        out.append(Band(b.name, cursor, upper, b.rate))
        cursor = upper if upper is not None else cursor
        if i == len(bands) - 1:
            break
    return out


def _dividend_bands(ty: TaxYear, extension: Decimal) -> list[Band]:
    uk = ty.income_tax_bands["england_ni"]
    basic_top = (uk[0].upper or ZERO) + extension
    higher_top = (uk[1].upper or ZERO) + extension
    return [
        Band("dividend_basic", ZERO, basic_top, ty.dividend_rates["basic"]),
        Band("dividend_higher", basic_top, higher_top, ty.dividend_rates["higher"]),
        Band("dividend_additional", higher_top, None, ty.dividend_rates["additional"]),
    ]


def _marginal_rate(tp: Taxpayer, delta: Decimal = D("100")) -> Decimal:
    """Empirical marginal rate: what fraction of the next £100 is lost.

    Computed by perturbation rather than by reasoning about bands, so it
    automatically captures the 60% taper zone and the HICBC spike without
    special-casing. This is the figure models get wrong most spectacularly.
    """
    base = compute_simple(tp)
    bumped = compute_simple(
        Taxpayer(
            salary=tp.salary + delta,
            salary_sacrifice=tp.salary_sacrifice,
            pension_relief_at_source=tp.pension_relief_at_source,
            pension_net_pay=tp.pension_net_pay,
            savings_interest=tp.savings_interest,
            dividends=tp.dividends,
            other_income=tp.other_income,
            loan_plans=tp.loan_plans,
            children=tp.children,
            claims_child_benefit=tp.claims_child_benefit,
            region=tp.region,
            year=tp.year,
        )
    )
    return ((bumped - base) / delta).quantize(D("0.0001"), rounding=ROUND_HALF_UP)


def compute_simple(tp: Taxpayer) -> Decimal:
    """Total deductions only — used by the marginal rate probe to avoid
    infinite recursion through compute()."""
    ty = get_year(tp.year)
    gross = max(ZERO, tp.salary - tp.salary_sacrifice)
    earnings = gross - tp.pension_net_pay + tp.other_income
    ani = adjusted_net_income(tp, ty)
    pa, _ = personal_allowance(ani, ty)

    remaining_pa = pa
    taxable_earnings = max(ZERO, earnings - remaining_pa)
    remaining_pa = max(ZERO, remaining_pa - earnings)
    taxable_savings_pre = max(ZERO, tp.savings_interest - remaining_pa)
    remaining_pa = max(ZERO, remaining_pa - tp.savings_interest)
    taxable_dividends_pre = max(ZERO, tp.dividends - remaining_pa)

    band_extension = tp.pension_relief_at_source * D("1.25")
    bands = list(ty.income_tax_bands[tp.region])
    if band_extension > ZERO:
        bands = _extend_bands(bands, band_extension)

    tax_earn, _, used = _tax_bands(taxable_earnings, bands, ZERO)
    uk_bands = list(ty.income_tax_bands["england_ni"])
    if band_extension > ZERO:
        uk_bands = _extend_bands(uk_bands, band_extension)
    total_taxable = taxable_earnings + taxable_savings_pre + taxable_dividends_pre
    zero_rated, taxable_savings = _savings_zero_rate(
        taxable_earnings, taxable_savings_pre, total_taxable, ty
    )
    tax_sav, _, used = _tax_bands(taxable_savings, uk_bands, used + zero_rated)
    div_allow = min(ty.dividend_allowance, taxable_dividends_pre)
    tax_div, _, _ = _tax_bands(
        taxable_dividends_pre - div_allow,
        _dividend_bands(ty, band_extension),
        used + div_allow,
    )

    ni = national_insurance(tp, ty)
    sl, _ = student_loan(tp, ty)
    _, charge = hicbc(tp, ani, ty)
    return _p(tax_earn + tax_sav + tax_div + ni + sl + charge)
