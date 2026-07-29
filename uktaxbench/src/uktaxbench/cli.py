"""Command line interface: generate, run, grade, report, verify-rates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import generator, runner
from .grading import grade_one
from .rates import CURRENT_YEAR, YEARS, unverified_years
from .report import build_report, leaderboard_markdown, write_report


def cmd_generate(args: argparse.Namespace) -> int:
    cases = generator.generate(seed=args.seed, year_pairs=not args.no_pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for c in cases:
            fh.write(json.dumps(c.to_json()) + "\n")
    print(f"Wrote {len(cases)} cases to {out}")
    return 0


def _load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    generator.register(cases)  # so the mock provider can answer them
    return cases


def cmd_run(args: argparse.Namespace) -> int:
    cases = _load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]

    all_grades = []
    for model in args.models:
        print(f"Running {model} over {len(cases)} cases")
        responses = runner.run_all(
            model, cases, temperature=args.temperature, use_cache=not args.no_cache
        )
        by_id = {c["id"]: c for c in cases}
        errs = 0
        for r in responses:
            if r.error:
                errs += 1
                continue
            all_grades.append(grade_one(by_id[r.case_id], r.text, model))
        if errs:
            print(f"  {errs} calls failed after retries", file=sys.stderr)

    meta = {"temperature": args.temperature, "models": args.models}
    full = build_report(all_grades, cases, meta=meta)
    write_report(full, Path(args.full_out))
    report = build_report(all_grades, cases, meta=meta, max_grades_per_model=300)
    path = write_report(report, Path(args.out))
    print(f"\nWrote dashboard report to {path}")
    print(f"Wrote full grades to {args.full_out}\n")
    print(leaderboard_markdown(report))
    return 0


def cmd_verify_rates(args: argparse.Namespace) -> int:
    bad = unverified_years()
    for y, ty in YEARS.items():
        mark = "ok" if ty.verified else "UNVERIFIED"
        print(f"{y}: {mark}  ({len(ty.sources)} sources listed)")
    if bad:
        print(
            "\nUnverified tax years: "
            + ", ".join(bad)
            + "\nCheck each figure against the gov.uk pages in rates.py, then set "
            "verified=True. Do not publish results until this passes.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="uktaxbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build the case set")
    g.add_argument("--out", default="data/cases.jsonl")
    g.add_argument("--seed", type=int, default=20252026)
    g.add_argument("--no-pairs", action="store_true", help="skip matched year twins")
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("run", help="run models and grade")
    r.add_argument("--cases", default="data/cases.jsonl")
    r.add_argument("--models", nargs="+", default=["mock-good", "mock-mid", "mock-poor"])
    r.add_argument("--out", default="web/results.json")
    r.add_argument("--full-out", default="data/grades_full.json")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--no-cache", action="store_true")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("verify-rates", help="check rate tables are verified")
    v.set_defaults(fn=cmd_verify_rates)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
