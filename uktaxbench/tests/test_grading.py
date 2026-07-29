"""Tests for extraction and error attribution.

The grader is the part most likely to be quietly wrong, because a bug here
shows up as a plausible-looking number rather than a crash. If extraction
silently fails, every model looks bad; if the hedge detector over-fires,
silent-error rates collapse and the headline finding evaporates.
"""

from decimal import Decimal as D

import pytest

from uktaxbench.generator import generate, year_twin
from uktaxbench.grading import (
    Grade,
    build_hypotheses,
    extract_answer,
    grade_one,
    summarise,
)
from uktaxbench.rates import CURRENT_YEAR, PRIOR_YEAR, figures_that_changed
from uktaxbench.reference import Taxpayer, compute


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<answer>1234.56</answer>", D("1234.56")),
        ("blah blah\n<answer>7890</answer>", D("7890")),
        ("Final answer: £4,486.00", D("4486.00")),
        ("The tax is £4,486.00 for the year.", D("4486.00")),
        ("First £1,000 then total £4,486.00", D("4486.00")),  # takes the last
        ("<answer>£1,794.40</answer>", D("1794.40")),
        ("Answer: 505 per year", D("505")),
    ],
)
def test_extracts_number(text, expected):
    assert extract_answer(text).value == expected


def test_unparseable_is_none_not_zero():
    """A missing answer must not be scored as 0, which could be correct."""
    e = extract_answer("I would need more information about their circumstances.")
    assert e.value is None
    assert e.method == "none"


def test_rate_percentages_normalise():
    assert extract_answer("<answer>62</answer>", unit="rate").value == D("0.62")
    assert extract_answer("<answer>0.62</answer>", unit="rate").value == D("0.62")


@pytest.mark.parametrize(
    "text",
    [
        "This is approximately £4,486.",
        "Roughly £4,486 — please verify with HMRC.",
        "I think the answer is £4,486.",
        "Around £4,486, though this assumes no other income.",
        "£4,486. You should consult a tax adviser.",
    ],
)
def test_hedges_detected(text):
    assert extract_answer(text).hedged is True


def test_confident_answer_not_hedged():
    assert extract_answer("The income tax due is £4,486.00.").hedged is False


def test_abstention_distinguished_from_wrong_answer():
    e = extract_answer("I cannot answer without knowing the tax year.")
    assert e.abstained is True


# --------------------------------------------------------------------------
# Silent error semantics
# --------------------------------------------------------------------------


def _case(answer, **kw):
    base = {
        "id": "test01",
        "tier": "L2_single",
        "answer": float(answer),
        "answer_field": "income_tax",
        "unit": "gbp",
        "tolerance": 1.0,
        "rules": [],
        "taxpayer": {"salary": 35000.0, "year": PRIOR_YEAR},
        "year": PRIOR_YEAR,
    }
    base.update(kw)
    return base


def test_correct_answer_is_not_a_silent_error():
    g = grade_one(_case(4486), "<answer>4486</answer>", "m")
    assert g.correct and not g.silent_error


def test_wrong_and_confident_is_silent_error():
    g = grade_one(_case(4486), "<answer>5000</answer>", "m")
    assert not g.correct and g.silent_error


def test_wrong_but_hedged_is_not_silent():
    g = grade_one(_case(4486), "Approximately <answer>5000</answer>", "m")
    assert not g.correct and not g.silent_error


def test_tolerance_is_one_pound():
    assert grade_one(_case(4486), "<answer>4486.99</answer>", "m").correct
    assert not grade_one(_case(4486), "<answer>4488</answer>", "m").correct


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def test_stale_year_attribution_fires():
    """A model answering a 2026/27 loan question with 2025/26 thresholds."""
    tp_new = Taxpayer(salary=D("35000"), loan_plans=("plan2",), year=CURRENT_YEAR)
    tp_old = Taxpayer(salary=D("35000"), loan_plans=("plan2",), year=PRIOR_YEAR)
    right = compute(tp_new).student_loan
    stale = compute(tp_old).student_loan
    assert right != stale  # the pair is discriminating

    case = _case(
        right,
        answer_field="student_loan",
        year=CURRENT_YEAR,
        taxpayer={"salary": 35000.0, "loan_plans": ["plan2"], "year": CURRENT_YEAR},
    )
    g = grade_one(case, f"<answer>{stale}</answer>", "m")
    assert not g.correct
    assert "stale_year" in g.attributions


def test_ignored_student_loan_attribution_fires():
    tp = Taxpayer(salary=D("40000"), loan_plans=("plan2",), year=CURRENT_YEAR)
    with_loan = compute(tp).take_home
    without = compute(Taxpayer(salary=D("40000"), year=CURRENT_YEAR)).take_home
    case = _case(
        with_loan,
        answer_field="take_home",
        year=CURRENT_YEAR,
        taxpayer={"salary": 40000.0, "loan_plans": ["plan2"], "year": CURRENT_YEAR},
    )
    g = grade_one(case, f"<answer>{without}</answer>", "m")
    assert "ignored_student_loan" in g.attributions


def test_no_attribution_for_random_wrong_number():
    g = grade_one(_case(4486), "<answer>9999.37</answer>", "m")
    assert g.attributions == []


def test_hypotheses_return_none_when_inapplicable():
    """A taxpayer with no loan cannot have 'ignored the loan' as an error."""
    hyps = build_hypotheses("income_tax")
    tp = Taxpayer(salary=D("30000"), year=CURRENT_YEAR)
    assert hyps["ignored_student_loan"](tp) is None
    assert hyps["scotland_as_england"](tp) is None
    assert hyps["no_hicbc"](tp) is None


# --------------------------------------------------------------------------
# Year pairing
# --------------------------------------------------------------------------


def test_only_loan_thresholds_moved_between_years():
    """Guards the benchmark's central design assumption.

    If a future year changes income tax too, the matched-pair analysis stops
    isolating one variable and this test should fail loudly.
    """
    changed = figures_that_changed(PRIOR_YEAR, CURRENT_YEAR)
    assert changed, "expected at least one difference between the two years"
    assert all(k.startswith("loan_threshold_") for k in changed), changed


def test_year_twin_preserves_scenario():
    cases = generate(mix={"L2_single": 20}, year_pairs=False)
    loan_cases = [c for c in cases if c.taxpayer.get("loan_plans")]
    if not loan_cases:
        pytest.skip("no loan cases in sample")
    c = loan_cases[0]
    t = year_twin(c)
    assert t is not None
    assert t.year == CURRENT_YEAR
    assert t.taxpayer["salary"] == c.taxpayer["salary"]
    assert PRIOR_YEAR not in t.prompt


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_summarise_rates_are_shares_not_counts():
    gs = [
        Grade("a", "L2_single", "m", True, 1.0, 1.0, 0.0, False, False, "tag", False),
        Grade("b", "L2_single", "m", False, 2.0, 1.0, 1.0, False, False, "tag", True),
    ]
    s = summarise(gs)["models"]["m"]
    assert s["accuracy"] == 0.5
    assert s["silent_error_rate"] == 0.5
    assert s["silent_share_of_errors"] == 1.0
