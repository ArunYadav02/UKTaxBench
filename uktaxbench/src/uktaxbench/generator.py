"""
Case generation.

Design notes:

  * Inputs are drawn from a seeded RNG at odd, non-round values (£47,318 not
    £47,000). Round salaries appear all over the training-data internet with
    worked answers attached; odd ones do not. This is the main contamination
    defence and it costs nothing.
  * Every case carries the exact rules it engages, taken from the reference
    implementation's own `rules_engaged` output rather than assumed by the
    generator. Per-rule accuracy is only meaningful if that labelling is
    trustworthy.
  * Regeneration with the same seed must reproduce the set byte for byte,
    otherwise published numbers cannot be checked by anyone else.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Callable, Iterable, Iterator

from .rates import TaxYear, get_year
from .reference import Taxpayer, compute

D = Decimal

TIERS = ("L1_lookup", "L2_single", "L3_interaction", "L4_adversarial")

# Fields a question can ask about, mapped to the TaxResult attribute.
ASKABLE = {
    "income_tax": "income tax",
    "national_insurance": "National Insurance",
    "student_loan": "student loan repayment",
    "take_home": "take-home pay",
    "personal_allowance": "personal allowance",
    "total_deductions": "total deductions",
    "hicbc": "High Income Child Benefit Charge",
    "marginal_rate": "marginal rate",
    "adjusted_net_income": "adjusted net income",
}


@dataclass
class Case:
    id: str
    tier: str
    prompt: str
    answer: float
    answer_field: str
    unit: str  # "gbp" or "rate"
    rules: list[str]
    taxpayer: dict[str, Any]
    year: str
    tolerance: float = 1.0
    origin: str = "generated"  # or "handwritten"
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _case_id(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _odd(rng: random.Random, low: int, high: int) -> Decimal:
    """A non-round amount in [low, high], to the pound.

    Deliberately avoids multiples of 500 — those are the values that appear in
    published worked examples and salary-guide tables.
    """
    for _ in range(50):
        v = rng.randint(low, high)
        if v % 500 != 0:
            return D(v)
    return D(low + 137)


def _fmt(x: Decimal) -> str:
    return f"£{x:,.0f}" if x == x.to_integral_value() else f"£{x:,.2f}"


def _build(
    tier: str,
    prompt: str,
    tp: Taxpayer,
    field_name: str,
    note: str = "",
    origin: str = "generated",
    tolerance: float = 1.0,
) -> Case:
    result = compute(tp)
    value = getattr(result, field_name)
    unit = "rate" if field_name == "marginal_rate" else "gbp"
    tp_dict = {
        k: (float(v) if isinstance(v, Decimal) else v)
        for k, v in asdict(tp).items()
    }
    payload = {"prompt": prompt, "taxpayer": tp_dict, "field": field_name}
    return Case(
        id=_case_id(payload),
        tier=tier,
        prompt=prompt,
        answer=float(value),
        answer_field=field_name,
        unit=unit,
        rules=result.rules_engaged or ["base_rates"],
        taxpayer=tp_dict,
        year=tp.year,
        tolerance=0.005 if unit == "rate" else tolerance,
        origin=origin,
        note=note,
    )


REGION_LABEL = {
    "england_ni": "England",
    "wales": "Wales",
    "scotland": "Scotland",
}

PLAN_LABEL = {
    "plan1": "Plan 1",
    "plan2": "Plan 2",
    "plan4": "Plan 4",
    "plan5": "Plan 5",
    "pgl": "a Postgraduate Loan",
}


# --------------------------------------------------------------------------
# L1 — rule lookup. No arithmetic, just: does the model know the thresholds?
# --------------------------------------------------------------------------


def gen_l1(rng: random.Random, n: int) -> Iterator[Case]:
    ty = get_year("2025/26")
    facts: list[tuple[str, Decimal, str]] = [
        (
            "For the 2025/26 UK tax year, what is the standard personal allowance? "
            "Answer with a single figure in pounds.",
            ty.personal_allowance,
            "personal_allowance_value",
        ),
        (
            "For 2025/26, at what level of adjusted net income does the personal "
            "allowance reduce to nil? Answer with a single figure in pounds.",
            D("125140"),
            "taper_end",
        ),
        (
            "For 2025/26, what is the Class 1 employee National Insurance upper "
            "earnings limit? Answer with a single figure in pounds.",
            ty.ni_upper_earnings_limit,
            "ni_uel",
        ),
        (
            "For 2025/26, what is the annual repayment threshold for a Plan 2 "
            "student loan? Answer with a single figure in pounds.",
            ty.loan_thresholds["plan2"],
            "plan2_threshold",
        ),
        (
            "For 2025/26, what is the annual repayment threshold for a Plan 5 "
            "student loan? Answer with a single figure in pounds.",
            ty.loan_thresholds["plan5"],
            "plan5_threshold",
        ),
        (
            "For 2025/26, what is the annual repayment threshold for a "
            "Postgraduate Loan? Answer with a single figure in pounds.",
            ty.loan_thresholds["pgl"],
            "pgl_threshold",
        ),
        (
            "For 2025/26, at what adjusted net income does the High Income Child "
            "Benefit Charge begin to apply? Answer with a single figure in pounds.",
            ty.hicbc_threshold,
            "hicbc_threshold",
        ),
        (
            "For 2025/26, at what adjusted net income does the High Income Child "
            "Benefit Charge claw back the full amount of Child Benefit? Answer "
            "with a single figure in pounds.",
            ty.hicbc_upper,
            "hicbc_upper",
        ),
        (
            "For 2025/26, what is the dividend allowance? Answer with a single "
            "figure in pounds.",
            ty.dividend_allowance,
            "dividend_allowance",
        ),
        (
            "For 2025/26, what is the personal savings allowance for a "
            "basic rate taxpayer? Answer with a single figure in pounds.",
            ty.psa_basic,
            "psa_basic",
        ),
        (
            "For 2025/26, what is the basic rate band width for income tax in "
            "England — that is, the amount of taxable income taxed at 20% before "
            "the higher rate begins? Answer with a single figure in pounds.",
            D("37700"),
            "basic_band_width",
        ),
        (
            "For 2025/26, what is the weekly rate of Child Benefit for the eldest "
            "child? Answer with a single figure in pounds.",
            ty.child_benefit_first,
            "cb_first",
        ),
        (
            "For 2025/26, at what gross income does the Scottish higher rate of "
            "42% begin? Answer with a single figure in pounds.",
            D("43663"),
            "scottish_higher_start",
        ),
        (
            "For 2025/26, what is the top rate of Scottish income tax, as a "
            "percentage? Answer with a single number.",
            D("48"),
            "scottish_top_rate",
        ),
    ]
    # The same fact is asked for both tax years. Income tax thresholds are
    # frozen across the two, but student loan thresholds are not — so this
    # pairing isolates models that answer "same as last year" by default.
    variants: list[tuple[str, Decimal, str, str]] = []
    for yr in ("2025/26", "2026/27"):
        tyy = get_year(yr)
        for prompt, answer, key in facts:
            p = prompt.replace("2025/26", yr)
            a = _fact_value(tyy, key, answer)
            variants.append((p, a, key, yr))

    for _ in range(n):
        prompt, answer, key, yr = rng.choice(variants)
        payload = {"prompt": prompt, "key": key, "year": yr}
        yield Case(
            id=_case_id(payload),
            tier="L1_lookup",
            prompt=prompt,
            answer=float(answer),
            answer_field=key,
            unit="rate" if "percentage" in prompt else "gbp",
            rules=["rate_lookup"],
            taxpayer={},
            year=yr,
            tolerance=0.0,
            note="Pure recall. Failures here are stale-year or hallucinated thresholds.",
        )


def _fact_value(ty: TaxYear, key: str, fallback: Decimal) -> Decimal:
    """Look the fact up in the given year's table rather than reusing 2025/26."""
    direct = {
        "personal_allowance_value": "personal_allowance",
        "ni_uel": "ni_upper_earnings_limit",
        "hicbc_threshold": "hicbc_threshold",
        "hicbc_upper": "hicbc_upper",
        "dividend_allowance": "dividend_allowance",
    }
    if key in direct:
        return getattr(ty, direct[key])
    if key.endswith("_threshold") and key.split("_")[0] in ty.loan_thresholds:
        return ty.loan_thresholds[key.split("_")[0]]
    return fallback


# --------------------------------------------------------------------------
# L2 — one rule, one calculation.
# --------------------------------------------------------------------------


def gen_l2(rng: random.Random, n: int) -> Iterator[Case]:
    for _ in range(n):
        kind = rng.choice(["tax", "ni", "loan", "takehome"])
        salary = _odd(rng, 13000, 145000)

        if kind == "tax":
            tp = Taxpayer(salary=salary)
            yield _build(
                "L2_single",
                f"An employee in England earns a gross salary of {_fmt(salary)} in "
                f"the 2025/26 tax year, with no other income, no pension "
                f"contributions and no student loan. How much income tax do they "
                f"pay for the year? Answer with a single figure in pounds.",
                tp,
                "income_tax",
            )
        elif kind == "ni":
            tp = Taxpayer(salary=salary)
            yield _build(
                "L2_single",
                f"An employee earns {_fmt(salary)} in 2025/26. How much Class 1 "
                f"employee National Insurance do they pay for the year? Assume "
                f"even monthly pay throughout. Answer with a single figure in "
                f"pounds.",
                tp,
                "national_insurance",
            )
        elif kind == "loan":
            plan = rng.choice(["plan1", "plan2", "plan4", "plan5", "pgl"])
            salary = _odd(rng, 26000, 90000)
            tp = Taxpayer(salary=salary, loan_plans=(plan,))
            yield _build(
                "L2_single",
                f"A graduate earning {_fmt(salary)} in 2025/26 is repaying "
                f"{PLAN_LABEL[plan]}. How much do they repay over the year? "
                f"Answer with a single figure in pounds.",
                tp,
                "student_loan",
            )
        else:
            tp = Taxpayer(salary=salary)
            yield _build(
                "L2_single",
                f"An employee in England earns {_fmt(salary)} in 2025/26 with no "
                f"other income, no pension and no student loan. What is their "
                f"annual take-home pay after income tax and National Insurance? "
                f"Answer with a single figure in pounds.",
                tp,
                "take_home",
            )


# --------------------------------------------------------------------------
# L3 — several rules interacting.
# --------------------------------------------------------------------------


def gen_l3(rng: random.Random, n: int) -> Iterator[Case]:
    for _ in range(n):
        kind = rng.choice(
            ["scotland", "two_loans", "taper", "sacrifice", "dividends", "hicbc"]
        )

        if kind == "scotland":
            salary = _odd(rng, 30000, 130000)
            tp = Taxpayer(salary=salary, region="scotland")
            yield _build(
                "L3_interaction",
                f"A Scottish taxpayer earns {_fmt(salary)} in 2025/26 with no "
                f"other income. How much income tax do they pay for the year? "
                f"Answer with a single figure in pounds.",
                tp,
                "income_tax",
                note="Requires Scottish bands, not UK bands.",
            )

        elif kind == "two_loans":
            salary = _odd(rng, 30000, 95000)
            tp = Taxpayer(salary=salary, loan_plans=("plan2", "pgl"))
            yield _build(
                "L3_interaction",
                f"Someone earning {_fmt(salary)} in 2025/26 is repaying both a "
                f"Plan 2 undergraduate loan and a Postgraduate Loan. What is "
                f"their total student loan repayment for the year? Answer with a "
                f"single figure in pounds.",
                tp,
                "student_loan",
                note="Both plans repay concurrently against separate thresholds.",
            )

        elif kind == "taper":
            salary = _odd(rng, 100500, 125000)
            tp = Taxpayer(salary=salary)
            field_choice = rng.choice(["personal_allowance", "income_tax"])
            asked = (
                "what is their personal allowance"
                if field_choice == "personal_allowance"
                else "how much income tax do they pay for the year"
            )
            yield _build(
                "L3_interaction",
                f"An employee in England earns {_fmt(salary)} in 2025/26 with no "
                f"other income and no pension contributions. Taking the personal "
                f"allowance taper into account, {asked}? Answer with a single "
                f"figure in pounds.",
                tp,
                field_choice,
                note="Allowance taper zone.",
            )

        elif kind == "sacrifice":
            salary = _odd(rng, 45000, 120000)
            sacrifice = _odd(rng, 3000, min(20000, int(salary) // 4))
            tp = Taxpayer(salary=salary, salary_sacrifice=sacrifice)
            yield _build(
                "L3_interaction",
                f"An employee in England has a gross salary of {_fmt(salary)} in "
                f"2025/26 and sacrifices {_fmt(sacrifice)} into their workplace "
                f"pension under a valid salary sacrifice arrangement. How much "
                f"Class 1 employee National Insurance do they pay for the year? "
                f"Answer with a single figure in pounds.",
                tp,
                "national_insurance",
                note="Sacrifice reduces NI-able pay.",
            )

        elif kind == "dividends":
            salary = _odd(rng, 20000, 80000)
            divs = _odd(rng, 2000, 30000)
            tp = Taxpayer(salary=salary, dividends=divs)
            yield _build(
                "L3_interaction",
                f"A director takes a salary of {_fmt(salary)} and {_fmt(divs)} in "
                f"UK dividends in 2025/26, living in England. What is their total "
                f"income tax for the year, including tax on the dividends? Answer "
                f"with a single figure in pounds.",
                tp,
                "income_tax",
                note="Dividends stack on top of earnings.",
            )

        else:  # hicbc
            salary = _odd(rng, 60500, 82000)
            children = rng.randint(1, 4)
            tp = Taxpayer(
                salary=salary, children=children, claims_child_benefit=True
            )
            yield _build(
                "L3_interaction",
                f"A parent in England earns {_fmt(salary)} in 2025/26 and their "
                f"household claims Child Benefit for {children} "
                f"{'child' if children == 1 else 'children'}. How much is the High "
                f"Income Child Benefit Charge for the year? Answer with a single "
                f"figure in pounds.",
                tp,
                "hicbc",
                note="Charge is stepped in whole 1% increments per £200.",
            )


# --------------------------------------------------------------------------
# L4 — adversarial. Traps that reward reasoning and punish pattern-matching.
# --------------------------------------------------------------------------


def gen_l4(rng: random.Random, n: int) -> Iterator[Case]:
    for _ in range(n):
        kind = rng.choice(
            [
                "sixty_percent",
                "pension_restores_allowance",
                "scotland_dividends",
                "sacrifice_vs_relief",
                "everything",
                "starting_rate",
            ]
        )

        if kind == "sixty_percent":
            salary = _odd(rng, 101000, 124000)
            tp = Taxpayer(salary=salary)
            yield _build(
                "L4_adversarial",
                f"An employee in England earning {_fmt(salary)} in 2025/26 is "
                f"offered a £100 pay rise. Counting income tax and Class 1 "
                f"employee National Insurance, what fraction of that £100 do they "
                f"lose? Answer as a decimal between 0 and 1.",
                tp,
                "marginal_rate",
                note="60% income tax from allowance withdrawal, plus 2% NI = 0.62.",
            )

        elif kind == "pension_restores_allowance":
            salary = _odd(rng, 105000, 124000)
            contrib = _odd(rng, 4000, 16000)
            tp = Taxpayer(salary=salary, pension_relief_at_source=contrib)
            yield _build(
                "L4_adversarial",
                f"An employee in England earns {_fmt(salary)} in 2025/26 and pays "
                f"{_fmt(contrib)} into a personal pension from their net pay "
                f"(relief at source). What is their adjusted net income for the "
                f"year? Answer with a single figure in pounds.",
                tp,
                "adjusted_net_income",
                note="Contribution grosses up by 25% before reducing ANI.",
            )

        elif kind == "scotland_dividends":
            salary = _odd(rng, 40000, 90000)
            divs = _odd(rng, 5000, 25000)
            tp = Taxpayer(salary=salary, dividends=divs, region="scotland")
            yield _build(
                "L4_adversarial",
                f"A Scottish taxpayer has employment income of {_fmt(salary)} and "
                f"{_fmt(divs)} of UK dividends in 2025/26. What is their total "
                f"income tax for the year? Answer with a single figure in pounds.",
                tp,
                "income_tax",
                note="Scottish rates on earnings, UK rates on dividends.",
            )

        elif kind == "sacrifice_vs_relief":
            salary = _odd(rng, 55000, 95000)
            amount = _odd(rng, 4000, 12000)
            tp = Taxpayer(salary=salary, pension_relief_at_source=amount)
            yield _build(
                "L4_adversarial",
                f"An employee in England earns {_fmt(salary)} in 2025/26 and pays "
                f"{_fmt(amount)} into a personal pension from net pay under "
                f"relief at source. How much Class 1 employee National Insurance "
                f"do they pay for the year? Answer with a single figure in pounds.",
                tp,
                "national_insurance",
                note="Relief at source does not reduce NI. Answer equals the "
                "no-pension figure.",
            )

        elif kind == "starting_rate":
            salary = _odd(rng, 12600, 17000)
            interest = _odd(rng, 3000, 9000)
            tp = Taxpayer(salary=salary, savings_interest=interest)
            yield _build(
                "L4_adversarial",
                f"Someone in England has employment income of {_fmt(salary)} and "
                f"{_fmt(interest)} of bank interest in 2025/26. How much income "
                f"tax do they pay in total for the year? Answer with a single "
                f"figure in pounds.",
                tp,
                "income_tax",
                note="Starting rate for savings and PSA both apply, in that order.",
            )

        else:  # everything
            salary = _odd(rng, 62000, 118000)
            children = rng.randint(1, 3)
            sacrifice = _odd(rng, 2000, 9000)
            tp = Taxpayer(
                salary=salary,
                salary_sacrifice=sacrifice,
                loan_plans=("plan2", "pgl"),
                children=children,
                claims_child_benefit=True,
            )
            yield _build(
                "L4_adversarial",
                f"An employee in England has a gross salary of {_fmt(salary)} in "
                f"2025/26 and sacrifices {_fmt(sacrifice)} into a workplace "
                f"pension. They repay both a Plan 2 loan and a Postgraduate Loan, "
                f"and their household claims Child Benefit for {children} "
                f"{'child' if children == 1 else 'children'}. What are their total "
                f"deductions for the year, counting income tax, National "
                f"Insurance, student loan repayments and any High Income Child "
                f"Benefit Charge? Answer with a single figure in pounds.",
                tp,
                "total_deductions",
                note="Five interacting rules. Sacrifice moves ANI, which moves HICBC.",
            )


GENERATORS: dict[str, Callable[[random.Random, int], Iterator[Case]]] = {
    "L1_lookup": gen_l1,
    "L2_single": gen_l2,
    "L3_interaction": gen_l3,
    "L4_adversarial": gen_l4,
}

DEFAULT_MIX = {
    # Post-pairing totals are roughly double these, since most cases
    # get a matched twin in the other tax year.
    "L1_lookup": 40,      # finite set; exhausts at 28
    "L2_single": 250,
    "L3_interaction": 300,
    "L4_adversarial": 160,
}


# Prompt -> case payload. Populated by generate(); the mock provider uses it
# to produce plausible answers without needing the case passed through the
# provider signature. Real providers never touch this.
CASES_BY_PROMPT: dict[str, dict[str, Any]] = {}


def year_twin(case: Case, to_year: str = "2026/27") -> Case | None:
    """Re-ask an identical scenario for a different tax year.

    Produces matched pairs: same salary, same plans, same wording, only the
    year differs. Because income tax thresholds are frozen across 2025/26 and
    2026/27 while student loan thresholds rose, a matched pair isolates one
    thing — whether the model tracked the change that actually happened.

    A model that gets both halves of a loan pair identical is not calculating,
    it is recalling. That is a much stronger claim than a raw accuracy gap,
    and it is only available because the pairs are matched.
    """
    if not case.taxpayer or case.year == to_year:
        return None
    from_year = case.year
    if from_year not in case.prompt:
        return None

    tp_data = dict(case.taxpayer)
    tp_data["year"] = to_year
    money = (
        "salary", "salary_sacrifice", "pension_relief_at_source",
        "pension_net_pay", "savings_interest", "dividends", "other_income",
    )
    kw: dict[str, Any] = {}
    for k, v in tp_data.items():
        if k in money:
            kw[k] = Decimal(str(v))
        elif k == "loan_plans":
            kw[k] = tuple(v)
        else:
            kw[k] = v
    try:
        tp = Taxpayer(**kw)
    except Exception:
        return None

    return _build(
        case.tier,
        case.prompt.replace(from_year, to_year),
        tp,
        case.answer_field,
        note=f"Year twin of {case.id} ({from_year}).",
        origin=case.origin,
        tolerance=case.tolerance,
    )


def generate(
    seed: int = 20252026,
    mix: dict[str, int] | None = None,
    year_pairs: bool = True,
) -> list[Case]:
    mix = mix or DEFAULT_MIX
    rng = random.Random(seed)
    cases: list[Case] = []
    seen: set[str] = set()
    for tier, count in mix.items():
        produced = 0
        stale = 0  # consecutive attempts yielding nothing new
        while produced < count and stale < 400:
            for case in GENERATORS[tier](rng, 1):
                if case.id in seen:
                    stale += 1
                    continue
                stale = 0
                seen.add(case.id)
                cases.append(case)
                produced += 1
                if produced >= count:
                    break
        if produced < count:
            # Honest reporting beats silently returning a short set: some
            # tiers (notably L1) have a finite number of distinct questions.
            print(
                f"  note: {tier} exhausted at {produced}/{count} distinct cases",
                flush=True,
            )

    if year_pairs:
        twins: list[Case] = []
        for c in cases:
            t = year_twin(c)
            if t is not None and t.id not in seen:
                seen.add(t.id)
                twins.append(t)
        cases.extend(twins)

    register(c.to_json() for c in cases)
    return cases


def register(cases: Iterable[dict[str, Any]]) -> None:
    """Index cases by prompt so the mock provider can answer them."""
    for c in cases:
        CASES_BY_PROMPT[c["prompt"]] = c
