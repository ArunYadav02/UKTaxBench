"""
Grading, and — more interestingly — error attribution.

Marking an answer right or wrong is easy and not very informative. "GPT-X
scores 71%" tells a reader nothing they can act on. This module answers the
follow-up question: *when the model is wrong, what did it actually do?*

The method is counterfactual recomputation. We re-run the reference
implementation under a set of deliberately broken hypotheses — last year's
thresholds, the personal allowance taper switched off, the student loan
ignored, Scotland treated as England — and check whether the model's number
matches one of them. If it does, we can name the mistake instead of just
recording a miss.

This is the part of the benchmark that produces sentences worth quoting:
not "23% error rate" but "in 41% of its errors on Scottish taxpayers, the
model applied English bands."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .rates import PRIOR_YEAR, get_year
from .reference import Taxpayer, compute

D = Decimal


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------

# Ordered by confidence. We try the explicit tagged form first because the
# prompt asks for it; free-text fallbacks exist because models ignore format
# instructions at a measurable rate, and "failed to follow format" is itself
# a result worth reporting rather than silently scoring as wrong.
_ANSWER_TAG = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.I | re.S)
_FINAL_LINE = re.compile(
    r"(?:final answer|answer)\s*[:\-]\s*(.+?)(?:\n|$)", re.I
)
_MONEY = re.compile(r"£\s*(-?[\d,]+(?:\.\d{1,2})?)")
_BARE_NUM = re.compile(r"(-?[\d,]+(?:\.\d{1,2})?)\s*%?")

_HEDGE_PATTERNS = [
    r"\bapproximat", r"\broughly\b", r"\baround\b", r"\bestimat", r"\babout\b",
    r"\bshould be\b", r"\bi think\b", r"\bnot (?:entirely )?(?:sure|certain)\b",
    r"\bmay (?:not )?be\b", r"\bmight\b", r"\bcheck with\b", r"\bverify\b",
    r"\bconsult\b", r"\bcould be\b", r"\bassum", r"\bdepend", r"\bcaveat",
    r"\bi (?:do not|don't) have\b", r"\buncertain", r"\bballpark\b",
    r"\bin the region of\b", r"\bsubject to\b", r"\bprofessional advice\b",
]
_HEDGE = re.compile("|".join(_HEDGE_PATTERNS), re.I)

# A refusal or explicit inability is NOT the same failure as a confident wrong
# answer, and conflating them would flatter models that simply decline.
_ABSTAIN = re.compile(
    r"\b(?:i (?:cannot|can't|am unable to)|unable to (?:answer|determine|calculate)"
    r"|insufficient information|not enough information)\b",
    re.I,
)


@dataclass
class Extraction:
    value: Decimal | None
    method: str          # how we found it: tag | final_line | money | bare | none
    hedged: bool
    abstained: bool
    raw: str

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["value"] = None if self.value is None else float(self.value)
        return d


def _to_decimal(s: str) -> Decimal | None:
    s = s.strip().replace(",", "").replace("£", "").replace("%", "").strip()
    # "1234.56 per year" -> take the leading number
    m = re.match(r"^(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return D(m.group(1))
    except InvalidOperation:
        return None


def extract_answer(text: str, unit: str = "gbp") -> Extraction:
    """Pull a single number out of a model response.

    Deliberately generous. A benchmark that scores zero because the model
    wrote "£1,234.00 per year" instead of "1234" is measuring instruction
    formatting, not tax knowledge. We record *how* we extracted it so that
    format-following can be reported separately.
    """
    hedged = bool(_HEDGE.search(text))
    abstained = bool(_ABSTAIN.search(text))

    def done(v: Decimal | None, method: str) -> Extraction:
        # "62%" and "0.62" are the same marginal rate. Normalise once, here,
        # so every extraction path agrees — an earlier version normalised only
        # in the fallback branch, which silently marked every correctly-stated
        # percentage wrong.
        if v is not None and unit == "rate" and v > 1:
            v = v / D(100)
        return Extraction(v, method, hedged, abstained, text)

    for pat, name in ((_ANSWER_TAG, "tag"), (_FINAL_LINE, "final_line")):
        m = pat.search(text)
        if m:
            v = _to_decimal(m.group(1))
            if v is not None:
                return done(v, name)

    # Fall back to the last money figure in the response — models tend to
    # state the final number last after showing working.
    money = _MONEY.findall(text)
    if money:
        v = _to_decimal(money[-1])
        if v is not None:
            return done(v, "money")

    if unit == "rate":
        nums = _BARE_NUM.findall(text)
        if nums:
            v = _to_decimal(nums[-1])
            if v is not None:
                return done(v, "bare")

    return Extraction(None, "none", hedged, abstained, text)


# --------------------------------------------------------------------------
# Counterfactual hypotheses
# --------------------------------------------------------------------------

Hypothesis = Callable[[Taxpayer], Decimal | None]


def _field_of(tp: Taxpayer, field_name: str) -> Decimal | None:
    try:
        res = compute(tp)
    except Exception:
        return None
    val = getattr(res, field_name, None)
    return val if isinstance(val, Decimal) else None


def _replace(tp: Taxpayer, **kw: Any) -> Taxpayer:
    data = {**tp.__dict__, **kw}
    return Taxpayer(**data)


def build_hypotheses(field_name: str) -> dict[str, Hypothesis]:
    """Named wrong-but-plausible ways to compute the same figure.

    Each returns the value a model would produce if it made exactly that one
    mistake. Order matters only for reporting; we test all and record every
    match, because some hypotheses coincide at some incomes.
    """

    def stale_year(tp: Taxpayer) -> Decimal | None:
        if tp.year == PRIOR_YEAR:
            return None
        return _field_of(_replace(tp, year=PRIOR_YEAR), field_name)

    def no_taper(tp: Taxpayer) -> Decimal | None:
        # Model forgot the personal allowance is withdrawn above £100k.
        # Simulated by moving the taper threshold out of reach.
        if tp.salary + tp.other_income + tp.dividends + tp.savings_interest <= D("100000"):
            return None
        return _hypothetical_no_taper(tp, field_name)

    def ignored_student_loan(tp: Taxpayer) -> Decimal | None:
        if not tp.loan_plans:
            return None
        return _field_of(_replace(tp, loan_plans=()), field_name)

    def scotland_as_england(tp: Taxpayer) -> Decimal | None:
        if tp.region != "scotland":
            return None
        return _field_of(_replace(tp, region="england_ni"), field_name)

    def england_as_scotland(tp: Taxpayer) -> Decimal | None:
        if tp.region == "scotland":
            return None
        return _field_of(_replace(tp, region="scotland"), field_name)

    def sacrifice_ignored(tp: Taxpayer) -> Decimal | None:
        if tp.salary_sacrifice == 0:
            return None
        return _field_of(_replace(tp, salary_sacrifice=D("0")), field_name)

    def sacrifice_reduces_tax_only(tp: Taxpayer) -> Decimal | None:
        # A common real-world confusion: treating salary sacrifice like a
        # net-pay pension contribution, so NI is not reduced.
        if tp.salary_sacrifice == 0:
            return None
        return _field_of(
            _replace(
                tp,
                salary_sacrifice=D("0"),
                pension_net_pay=tp.pension_net_pay + tp.salary_sacrifice,
            ),
            field_name,
        )

    def no_hicbc(tp: Taxpayer) -> Decimal | None:
        if not tp.claims_child_benefit:
            return None
        return _field_of(_replace(tp, claims_child_benefit=False, children=0), field_name)

    def pgl_at_nine_percent(tp: Taxpayer) -> Decimal | None:
        # Applying the 9% undergraduate rate to a postgraduate loan.
        if "pgl" not in tp.loan_plans:
            return None
        ty = get_year(tp.year)
        base = _field_of(tp, field_name)
        if base is None:
            return None
        income = tp.salary - tp.salary_sacrifice + tp.other_income
        over = max(D("0"), income - ty.loan_thresholds["pgl"])
        correct = (over * D("0.06")).to_integral_value(rounding="ROUND_DOWN")
        wrong = (over * D("0.09")).to_integral_value(rounding="ROUND_DOWN")
        delta = wrong - correct
        # student_loan and take_home move in opposite directions
        if field_name in ("student_loan", "total_deductions"):
            return base + delta
        if field_name == "take_home":
            return base - delta
        return None

    return {
        "stale_year": stale_year,
        "no_taper": no_taper,
        "ignored_student_loan": ignored_student_loan,
        "scotland_as_england": scotland_as_england,
        "england_as_scotland": england_as_scotland,
        "sacrifice_ignored": sacrifice_ignored,
        "sacrifice_reduces_tax_only": sacrifice_reduces_tax_only,
        "no_hicbc": no_hicbc,
        "pgl_at_nine_percent": pgl_at_nine_percent,
    }


def _hypothetical_no_taper(tp: Taxpayer, field_name: str) -> Decimal | None:
    """Recompute as if the personal allowance were never withdrawn.

    Implemented by temporarily swapping in a rate table whose taper threshold
    is unreachable, rather than by patching the reference implementation.
    """
    from dataclasses import replace as dc_replace
    from . import rates as rates_mod

    ty = get_year(tp.year)
    patched = dc_replace(ty, taper_threshold=D("999999999"))
    original = rates_mod.YEARS[tp.year]
    rates_mod.YEARS[tp.year] = patched
    try:
        return _field_of(tp, field_name)
    finally:
        rates_mod.YEARS[tp.year] = original


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------


@dataclass
class Grade:
    case_id: str
    tier: str
    model: str
    correct: bool
    parsed: float | None
    expected: float
    abs_error: float | None
    hedged: bool
    abstained: bool
    parse_method: str
    silent_error: bool           # wrong, and expressed no uncertainty
    attributions: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    year: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def grade_one(
    case: dict[str, Any],
    response_text: str,
    model: str,
) -> Grade:
    unit = case.get("unit", "gbp")
    ext = extract_answer(response_text, unit=unit)
    expected = D(str(case["answer"]))
    tol = D(str(case.get("tolerance", 1.0)))

    correct = ext.value is not None and abs(ext.value - expected) <= tol
    abs_err = None if ext.value is None else float(abs(ext.value - expected))

    attributions: list[str] = []
    if not correct and ext.value is not None:
        tp_data = dict(case["taxpayer"])
        try:
            tp = _rehydrate(tp_data)
        except Exception:
            tp = None
        if tp is not None:
            for name, hyp in build_hypotheses(case["answer_field"]).items():
                try:
                    alt = hyp(tp)
                except Exception:
                    alt = None
                if alt is not None and abs(ext.value - alt) <= tol and alt != expected:
                    attributions.append(name)

    return Grade(
        case_id=case["id"],
        tier=case["tier"],
        model=model,
        correct=correct,
        parsed=None if ext.value is None else float(ext.value),
        expected=float(expected),
        abs_error=abs_err,
        hedged=ext.hedged,
        abstained=ext.abstained,
        parse_method=ext.method,
        silent_error=(not correct) and (not ext.hedged) and (not ext.abstained),
        attributions=attributions,
        rules=list(case.get("rules", [])),
        year=case.get("year", ""),
    )


def _rehydrate(d: dict[str, Any]) -> Taxpayer:
    """Rebuild a Taxpayer from serialised JSON."""
    money = (
        "salary", "salary_sacrifice", "pension_relief_at_source",
        "pension_net_pay", "savings_interest", "dividends", "other_income",
    )
    kw: dict[str, Any] = {}
    for k, v in d.items():
        if k in money:
            kw[k] = D(str(v))
        elif k == "loan_plans":
            kw[k] = tuple(v)
        else:
            kw[k] = v
    return Taxpayer(**kw)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def summarise(grades: list[Grade]) -> dict[str, Any]:
    """Per-model headline metrics plus the breakdowns worth publishing."""
    by_model: dict[str, list[Grade]] = {}
    for g in grades:
        by_model.setdefault(g.model, []).append(g)

    out: dict[str, Any] = {"models": {}}
    for model, gs in by_model.items():
        n = len(gs)
        correct = sum(g.correct for g in gs)
        wrong = [g for g in gs if not g.correct]
        silent = [g for g in wrong if g.silent_error]
        answered = [g for g in gs if g.parsed is not None]

        tiers: dict[str, dict[str, float]] = {}
        for t in sorted({g.tier for g in gs}):
            sub = [g for g in gs if g.tier == t]
            tiers[t] = {
                "n": len(sub),
                "accuracy": round(sum(x.correct for x in sub) / len(sub), 4),
            }

        rules: dict[str, dict[str, float]] = {}
        for r in sorted({r for g in gs for r in g.rules}):
            sub = [g for g in gs if r in g.rules]
            if sub:
                rules[r] = {
                    "n": len(sub),
                    "accuracy": round(sum(x.correct for x in sub) / len(sub), 4),
                }

        attrib: dict[str, int] = {}
        for g in wrong:
            for a in g.attributions:
                attrib[a] = attrib.get(a, 0) + 1

        errs = [g.abs_error for g in answered if g.abs_error is not None]
        errs.sort()

        out["models"][model] = {
            "n": n,
            "accuracy": round(correct / n, 4) if n else 0.0,
            # The headline number of this benchmark. Of everything the model
            # got wrong, how much did it assert without any hedge?
            "silent_error_rate": round(len(silent) / n, 4) if n else 0.0,
            "silent_share_of_errors": round(len(silent) / len(wrong), 4) if wrong else 0.0,
            "hedge_rate": round(sum(g.hedged for g in gs) / n, 4) if n else 0.0,
            "abstain_rate": round(sum(g.abstained for g in gs) / n, 4) if n else 0.0,
            "unparseable_rate": round((n - len(answered)) / n, 4) if n else 0.0,
            "median_abs_error": round(errs[len(errs) // 2], 2) if errs else None,
            "p90_abs_error": round(errs[int(len(errs) * 0.9)], 2) if errs else None,
            "explained_error_share": (
                round(sum(1 for g in wrong if g.attributions) / len(wrong), 4)
                if wrong else 0.0
            ),
            "by_tier": tiers,
            "by_rule": rules,
            "error_attribution": dict(
                sorted(attrib.items(), key=lambda kv: -kv[1])
            ),
        }
    return out
