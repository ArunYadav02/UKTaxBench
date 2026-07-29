"""
Turning grades into the artifact people actually read.

Produces one JSON file consumed by the dashboard, plus a markdown leaderboard
for the README. The matched-pair analysis lives here because it needs the
whole grade set at once, not one case at a time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .grading import Grade, summarise
from .rates import CURRENT_YEAR, PRIOR_YEAR, figures_that_changed


def paired_analysis(
    grades: list[Grade], cases: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compare each model against itself across matched year twins.

    A pair is two cases with identical wording bar the tax year. We care about
    pairs where the correct answers differ — i.e. the ones a student loan
    threshold change actually moved. If a model gives the *same* number to
    both, it did not use the year at all.
    """
    # Index twins by their scenario signature: prompt with the year stripped.
    by_sig: dict[str, dict[str, str]] = {}
    for cid, c in cases.items():
        sig = c["prompt"].replace(c["year"], "<YEAR>")
        by_sig.setdefault(sig, {})[c["year"]] = cid

    pairs = [
        (v[PRIOR_YEAR], v[CURRENT_YEAR])
        for v in by_sig.values()
        if PRIOR_YEAR in v and CURRENT_YEAR in v
    ]

    out: dict[str, Any] = {
        "n_pairs": len(pairs),
        "changed_figures": {
            k: [float(a), float(b)]
            for k, (a, b) in figures_that_changed(PRIOR_YEAR, CURRENT_YEAR).items()
        },
        "models": {},
    }

    by_model_case: dict[str, dict[str, Grade]] = {}
    for g in grades:
        by_model_case.setdefault(g.model, {})[g.case_id] = g

    for model, gmap in by_model_case.items():
        discriminating = 0   # pairs where the right answers differ
        collapsed = 0        # model gave the same answer to both anyway
        both_right = 0
        for old_id, new_id in pairs:
            go, gn = gmap.get(old_id), gmap.get(new_id)
            if not go or not gn:
                continue
            if go.expected == gn.expected:
                continue  # not discriminating; the answer genuinely didn't move
            discriminating += 1
            if go.correct and gn.correct:
                both_right += 1
            if (
                go.parsed is not None
                and gn.parsed is not None
                and abs(go.parsed - gn.parsed) < 0.01
            ):
                collapsed += 1
        out["models"][model] = {
            "discriminating_pairs": discriminating,
            "answered_identically": collapsed,
            # The headline: share of year-sensitive scenarios where the model
            # produced one number regardless of which year was asked about.
            "year_blind_rate": (
                round(collapsed / discriminating, 4) if discriminating else None
            ),
            "both_correct_rate": (
                round(both_right / discriminating, 4) if discriminating else None
            ),
        }
    return out


def build_report(
    grades: list[Grade],
    cases: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    max_grades_per_model: int | None = None,
) -> dict[str, Any]:
    """Assemble the published artifact.

    `max_grades_per_model` trims the per-case detail for the dashboard. The
    full set is written separately: a 2 MB page load to show a leaderboard is
    a bad trade, but discarding the raw grades entirely would make the results
    unauditable. Errors are kept in preference to correct answers, since the
    detail view exists to show failures.
    """
    case_map = {c["id"]: c for c in cases}
    summary = summarise(grades)

    kept = grades
    if max_grades_per_model:
        kept = []
        by_model: dict[str, list[Grade]] = {}
        for g in grades:
            by_model.setdefault(g.model, []).append(g)
        for gs in by_model.values():
            wrong = [g for g in gs if not g.correct]
            right = [g for g in gs if g.correct]
            take = wrong[:max_grades_per_model]
            take += right[: max(0, max_grades_per_model - len(take))]
            kept.extend(take)

    # Carry the question text alongside each grade. A results file that shows
    # a discrepancy without the question it came from is not auditable, and
    # auditability is most of the value of publishing raw grades at all.
    out_grades = []
    for g in kept:
        d = g.to_json()
        c = case_map.get(g.case_id, {})
        d["prompt"] = c.get("prompt", "")
        d["note"] = c.get("note", "")
        out_grades.append(d)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "tax_years": [PRIOR_YEAR, CURRENT_YEAR],
        "meta": meta or {},
        "summary": summary,
        "paired": paired_analysis(grades, case_map),
        "grades": out_grades,
        "grades_truncated": bool(max_grades_per_model) and len(kept) < len(grades),
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def leaderboard_markdown(report: dict[str, Any]) -> str:
    """The table that goes in the README."""
    rows = []
    models = report["summary"]["models"]
    paired = report.get("paired", {}).get("models", {})
    for name, m in sorted(models.items(), key=lambda kv: -kv[1]["accuracy"]):
        yb = paired.get(name, {}).get("year_blind_rate")
        yb_s = "—" if yb is None else f"{yb:.1%}"
        med = m["median_abs_error"]
        med_s = "—" if med is None else f"£{med:,.0f}"
        rows.append(
            f"| {name} | {m['accuracy']:.1%} | {m['silent_error_rate']:.1%} | "
            f"{m['silent_share_of_errors']:.1%} | {yb_s} | {med_s} |"
        )
    header = (
        "| Model | Accuracy | Silent error rate | Silent share of errors | "
        "Year-blind rate | Median error |\n"
        "|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows)
