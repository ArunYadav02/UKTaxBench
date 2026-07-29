"""
Every expected value here is computed by hand from published thresholds and
written out longhand in the docstring or comment. If a test fails, work out
which of the two is wrong before changing either — silently editing the
expected value to match the code defeats the purpose of the file.
"""

from decimal import Decimal as D

import pytest

from uktaxbench.reference import (
    Taxpayer,
    compute,
    hicbc,
    national_insurance,
    personal_allowance,
    student_loan,
)
from uktaxbench.rates import get_year

TY = get_year("2025/26")


# --------------------------------------------------------------------------
# Income tax, England & NI
# --------------------------------------------------------------------------


def test_basic_rate_only():
    """£30,000 salary.
    Taxable: 30,000 - 12,570 = 17,430, all basic.
    Tax: 17,430 x 20% = 3,486.00
    """
    r = compute(Taxpayer(salary=D("30000")))
    assert r.income_tax == D("3486.00")


def test_below_personal_allowance_pays_nothing():
    r = compute(Taxpayer(salary=D("10000")))
    assert r.income_tax == D("0")
    assert r.national_insurance == D("0")


def test_exactly_at_personal_allowance():
    r = compute(Taxpayer(salary=D("12570")))
    assert r.income_tax == D("0")
    assert r.national_insurance == D("0")


def test_higher_rate():
    """£60,000 salary.
    Taxable: 47,430. Basic 37,700 x 20% = 7,540.
    Higher: 47,430 - 37,700 = 9,730 x 40% = 3,892.
    Total: 11,432.00
    """
    r = compute(Taxpayer(salary=D("60000")))
    assert r.income_tax == D("11432.00")


def test_additional_rate():
    """£150,000 salary. Allowance fully tapered away.
    Taxable: 150,000. Basic 37,700 x 20% = 7,540.
    Higher: 125,140 - 37,700 = 87,440 x 40% = 34,976.
    Additional: 150,000 - 125,140 = 24,860 x 45% = 11,187.
    Total: 53,703.00
    """
    r = compute(Taxpayer(salary=D("150000")))
    assert r.personal_allowance == D("0")
    assert r.income_tax == D("53703.00")


# --------------------------------------------------------------------------
# Personal allowance taper
# --------------------------------------------------------------------------


def test_taper_halfway():
    """ANI 110,000. Excess 10,000, allowance lost 5,000 -> PA 7,570."""
    pa, lost = personal_allowance(D("110000"), TY)
    assert pa == D("7570")
    assert lost == D("5000")


def test_taper_exhausted_at_125140():
    pa, lost = personal_allowance(D("125140"), TY)
    assert pa == D("0")
    assert lost == D("12570")


def test_taper_does_not_go_negative():
    pa, _ = personal_allowance(D("500000"), TY)
    assert pa == D("0")


def test_sixty_percent_marginal_band():
    """The signature UK oddity: 60% effective marginal income tax between
    100,000 and 125,140, because each extra £1 also removes 50p of allowance.
    With 2% NI on top the total marginal rate is 62%."""
    r = compute(Taxpayer(salary=D("110000")))
    assert r.marginal_rate == D("0.6200")


def test_marginal_rate_normal_higher_rate_is_42_percent():
    r = compute(Taxpayer(salary=D("70000")))
    assert r.marginal_rate == D("0.4200")


def test_marginal_rate_basic_is_28_percent():
    r = compute(Taxpayer(salary=D("30000")))
    assert r.marginal_rate == D("0.2800")


# --------------------------------------------------------------------------
# National Insurance
# --------------------------------------------------------------------------


def test_ni_basic():
    """£30,000: (30,000 - 12,570) x 8% = 17,430 x 8% = 1,394.40"""
    assert national_insurance(Taxpayer(salary=D("30000"))) == D("1394.40")


def test_ni_above_upper_earnings_limit():
    """£60,000: 37,700 x 8% = 3,016.00, plus 9,730 x 2% = 194.60 -> 3,210.60"""
    assert national_insurance(Taxpayer(salary=D("60000"))) == D("3210.60")


def test_ni_at_upper_earnings_limit_exactly():
    assert national_insurance(Taxpayer(salary=D("50270"))) == D("3016.00")


def test_salary_sacrifice_reduces_ni():
    """Sacrifice reduces NI-able pay. £60,000 sacrificing £10,000 pays the
    same NI as someone on £50,000."""
    sacrificed = national_insurance(
        Taxpayer(salary=D("60000"), salary_sacrifice=D("10000"))
    )
    assert sacrificed == national_insurance(Taxpayer(salary=D("50000")))


def test_relief_at_source_pension_does_not_reduce_ni():
    """The asymmetry that catches people out: a personal pension contribution
    cuts income tax but leaves NI untouched."""
    plain = national_insurance(Taxpayer(salary=D("60000")))
    with_pension = national_insurance(
        Taxpayer(salary=D("60000"), pension_relief_at_source=D("8000"))
    )
    assert plain == with_pension


def test_net_pay_pension_does_not_reduce_ni():
    plain = national_insurance(Taxpayer(salary=D("60000")))
    with_pension = national_insurance(
        Taxpayer(salary=D("60000"), pension_net_pay=D("5000"))
    )
    assert plain == with_pension


# --------------------------------------------------------------------------
# Student loans
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan,salary,expected",
    [
        # (salary - threshold) x rate, rounded down to whole pounds
        ("plan1", D("35000"), D("804")),   # (35,000-26,065) x 9% = 804.15 -> 804
        ("plan2", D("35000"), D("587")),   # (35,000-28,470) x 9% = 587.70 -> 587
        ("plan4", D("40000"), D("652")),   # (40,000-32,745) x 9% = 652.95 -> 652
        ("plan5", D("35000"), D("900")),   # (35,000-25,000) x 9% = 900.00 -> 900
        ("pgl", D("35000"), D("840")),     # (35,000-21,000) x 6% = 840.00 -> 840
    ],
)
def test_single_loan_plan(plan, salary, expected):
    total, _ = student_loan(Taxpayer(salary=salary, loan_plans=(plan,)))
    assert total == expected


def test_below_threshold_repays_nothing():
    total, _ = student_loan(Taxpayer(salary=D("20000"), loan_plans=("plan2",)))
    assert total == D("0")


def test_undergraduate_and_postgraduate_repay_simultaneously():
    """Both loans are repaid at once against separate thresholds.
    Plan 2: 587. PGL: 840. Total 1,427 — not the larger of the two."""
    total, by_plan = student_loan(
        Taxpayer(salary=D("35000"), loan_plans=("plan2", "pgl"))
    )
    assert by_plan["plan2"] == D("587")
    assert by_plan["pgl"] == D("840")
    assert total == D("1427")


def test_salary_sacrifice_reduces_loan_repayment():
    plain, _ = student_loan(Taxpayer(salary=D("40000"), loan_plans=("plan2",)))
    sacrificed, _ = student_loan(
        Taxpayer(salary=D("40000"), salary_sacrifice=D("5000"), loan_plans=("plan2",))
    )
    assert sacrificed < plain


# --------------------------------------------------------------------------
# High Income Child Benefit Charge
# --------------------------------------------------------------------------


def test_child_benefit_two_children():
    """(26.05 + 17.25) x 52 = 43.30 x 52 = 2,251.60"""
    r = compute(
        Taxpayer(salary=D("40000"), children=2, claims_child_benefit=True)
    )
    assert r.child_benefit_received == D("2251.60")
    assert r.hicbc == D("0")


def test_hicbc_halfway():
    """ANI 70,000 -> (70,000-60,000)/200 = 50 steps -> 50% of 2,251.60 = 1,125.80"""
    tp = Taxpayer(salary=D("70000"), children=2, claims_child_benefit=True)
    benefit, charge = hicbc(tp, D("70000"))
    assert benefit == D("2251.60")
    assert charge == D("1125.80")


def test_hicbc_full_clawback_at_80000():
    tp = Taxpayer(salary=D("85000"), children=2, claims_child_benefit=True)
    benefit, charge = hicbc(tp, D("85000"))
    assert charge == benefit


def test_hicbc_uses_adjusted_net_income_not_salary():
    """£70,000 salary with an £8,000 relief-at-source pension contribution
    grosses up to £10,000, giving ANI of £60,000 — exactly at the threshold,
    so the charge disappears entirely."""
    tp = Taxpayer(
        salary=D("70000"),
        pension_relief_at_source=D("8000"),
        children=2,
        claims_child_benefit=True,
    )
    r = compute(tp)
    assert r.adjusted_net_income == D("60000")
    assert r.hicbc == D("0")


def test_no_charge_if_not_claiming():
    r = compute(Taxpayer(salary=D("90000"), children=2, claims_child_benefit=False))
    assert r.hicbc == D("0")
    assert r.child_benefit_received == D("0")


# --------------------------------------------------------------------------
# Savings and dividends
# --------------------------------------------------------------------------


def test_psa_covers_small_interest_for_basic_rate_payer():
    r = compute(Taxpayer(salary=D("30000"), savings_interest=D("800")))
    assert r.tax_on_savings == D("0")


def test_psa_partially_covers_interest():
    """£30,000 salary, £1,500 interest. PSA 1,000, so 500 taxed at 20% = 100."""
    r = compute(Taxpayer(salary=D("30000"), savings_interest=D("1500")))
    assert r.tax_on_savings == D("100.00")


def test_higher_rate_payer_gets_smaller_psa():
    """£60,000 salary, £1,500 interest. PSA 500, so 1,000 taxed at 40% = 400."""
    r = compute(Taxpayer(salary=D("60000"), savings_interest=D("1500")))
    assert r.tax_on_savings == D("400.00")


def test_starting_rate_for_savings():
    """£14,000 salary, £6,000 interest.
    Taxable earnings 1,430, so starting rate band left = 5,000 - 1,430 = 3,570.
    Of 6,000 interest: 3,570 at 0%, then PSA 1,000 at 0%, leaving 1,430 at 20% = 286.
    """
    r = compute(Taxpayer(salary=D("14000"), savings_interest=D("6000")))
    assert r.tax_on_savings == D("286.00")


def test_dividend_allowance():
    r = compute(Taxpayer(salary=D("30000"), dividends=D("500")))
    assert r.tax_on_dividends == D("0")


def test_dividends_stack_above_salary():
    """£50,000 salary, £10,000 dividends. Allowance 500 uses band space, and
    the remaining 9,500 falls in the higher rate: 9,500 x 33.75% = 3,206.25"""
    r = compute(Taxpayer(salary=D("50000"), dividends=D("10000")))
    assert r.tax_on_dividends == D("3206.25")


# --------------------------------------------------------------------------
# Scotland
# --------------------------------------------------------------------------


def test_scotland_differs_from_england_at_50k():
    """£50,000 in Scotland. Taxable 37,430.
    Starter    2,827 x 19% =   537.13
    Basic     12,094 x 20% = 2,418.80
    Interm.   16,171 x 21% = 3,395.91
    Higher     6,338 x 42% = 2,661.96
    Total                  = 9,013.80
    versus 7,486.00 in England.
    """
    scot = compute(Taxpayer(salary=D("50000"), region="scotland"))
    eng = compute(Taxpayer(salary=D("50000"), region="england_ni"))
    assert scot.income_tax == D("9013.80")
    assert eng.income_tax == D("7486.00")
    assert scot.income_tax > eng.income_tax


def test_scottish_ni_is_identical():
    """NI is reserved, so it does not vary by nation."""
    scot = national_insurance(Taxpayer(salary=D("50000"), region="scotland"))
    eng = national_insurance(Taxpayer(salary=D("50000"), region="england_ni"))
    assert scot == eng


def test_scottish_dividends_use_uk_rates():
    """Dividend taxation is not devolved."""
    scot = compute(Taxpayer(salary=D("50000"), dividends=D("10000"), region="scotland"))
    eng = compute(Taxpayer(salary=D("50000"), dividends=D("10000")))
    assert scot.tax_on_dividends == eng.tax_on_dividends


def test_wales_matches_england():
    wales = compute(Taxpayer(salary=D("50000"), region="wales"))
    eng = compute(Taxpayer(salary=D("50000")))
    assert wales.income_tax == eng.income_tax


# --------------------------------------------------------------------------
# Consistency properties
# --------------------------------------------------------------------------


def test_take_home_reconciles():
    r = compute(Taxpayer(salary=D("45000"), loan_plans=("plan2",)))
    assert r.take_home == D("45000") - r.total_deductions


def test_tax_is_monotonic_in_salary():
    last = D("-1")
    for salary in range(0, 200000, 2500):
        r = compute(Taxpayer(salary=D(salary)))
        assert r.income_tax >= last
        last = r.income_tax


def test_take_home_never_decreases_with_salary():
    """No cliff edge should ever leave someone worse off for earning more.
    HICBC is a taper, not a cliff, so this must hold even for claimants."""
    last = D("-1")
    for salary in range(0, 200000, 500):
        r = compute(
            Taxpayer(salary=D(salary), children=2, claims_child_benefit=True)
        )
        assert r.take_home >= last, f"take-home fell at {salary}"
        last = r.take_home
