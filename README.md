# UKTaxBench

A benchmark for whether language models correctly apply UK personal tax rules —
income tax, National Insurance, student loan repayment, the personal allowance
taper, and the High Income Child Benefit Charge.

Ground truth comes from a reference implementation rather than hand-written
answer keys, so every case has one exactly verifiable number behind it and the
set can be regenerated at any size.

**This is a benchmark artifact. It is not tax advice.** See [Limitations](#limitations).

---

## The finding this is built to measure

Accuracy is the least interesting number here. Three things matter more:

**Silent error rate.** How often a model is wrong *and* expresses no
uncertainty — no hedge, no caveat, no suggestion to check with HMRC. A model
that is wrong 20% of the time but flags its own doubt is usable. A model that
is wrong 20% of the time with unbroken confidence is not.

**Year-blindness.** Income tax thresholds and NI rates are *frozen* between
2025/26 and 2026/27. Student loan thresholds are not: Plan 1 rose from £26,065
to £26,900, Plan 2 from £28,470 to £29,385, Plan 4 from £32,745 to £33,795,
while Plan 5 and postgraduate stayed put.

That asymmetry makes a controlled experiment possible. Each scenario is asked
twice — identical wording, only the tax year differs — giving matched pairs
where the correct answers genuinely diverge. A model returning the same figure
for both did not use the year at all. It recalled rather than calculated.

**Error attribution.** When a model misses, the harness re-runs the calculation
under nine deliberately broken hypotheses and reports which one reproduces the
model's number:

| Hypothesis | The mistake it represents |
|---|---|
| `stale_year` | Used the previous tax year's thresholds |
| `ignored_student_loan` | Left the repayment out entirely |
| `no_taper` | Didn't withdraw the personal allowance above £100,000 |
| `scotland_as_england` | Gave a Scottish taxpayer rest-of-UK bands |
| `england_as_scotland` | The reverse |
| `sacrifice_ignored` | Didn't deduct salary sacrifice from gross pay |
| `sacrifice_reduces_tax_only` | Reduced tax but not NI |
| `no_hicbc` | Skipped the High Income Child Benefit Charge |
| `pgl_at_nine_percent` | Repaid a postgraduate loan at 9% instead of 6% |

This turns "wrong" into "forgot the student loan," which is the difference
between a number and a finding.

---

## Results

Run `uktaxbench run` to populate. The table below is from the built-in mock
models — they exist to exercise the harness, and their scores mean nothing
about any real system.

| Model | Accuracy | Silent error rate | Silent share of errors | Year-blind | Median error |
|---|---|---|---|---|---|
| mock-good | 87.7% | 8.8% | 71.4% | 5.3% | £0 |
| mock-mid | 66.0% | 21.1% | 61.9% | 5.3% | £0 |
| mock-poor | 46.6% | 35.9% | 67.2% | 8.3% | £4 |

---

## Quick start

```bash
pip install -e ".[dev]"

python -m uktaxbench.cli generate                    # writes data/cases.jsonl
python -m uktaxbench.cli run                         # mock models, no API key, no spend
python -m http.server --directory web                # dashboard at :8000
```

Against real models:

```bash
export ANTHROPIC_API_KEY=...
python -m uktaxbench.cli run --models claude-sonnet-4-6 --limit 100
```

Start with `--limit`. Responses are cached on disk by `(model, prompt,
temperature)`, so re-runs while you fix the harness cost nothing.

---

## Case tiers

| Tier | What it tests | Count |
|---|---|---|
| **L1 lookup** | Recall a published threshold. No arithmetic. | 28 |
| **L2 single** | One rule, one calculation. | 500 |
| **L3 interaction** | Several rules on one taxpayer. | 600 |
| **L4 adversarial** | Taper zone, duplicate loan plans, regional divergence. | 320 |

1,448 total, split evenly across both tax years. L1 is deliberately small: it
is a finite set of published facts, and the generator reports exhaustion rather
than padding it with near-duplicates.

Salaries avoid round multiples of £500 — the values that appear in published
worked examples and salary guides — so cases are unlikely to appear verbatim in
training data.

---

## Layout

```
src/uktaxbench/
  rates.py        Rate tables. Every figure carries a source URL and verified flag.
  reference.py    Ground truth. Decimal throughout; floats corrupt money silently.
  generator.py    Case construction, tiers, matched year-pairs.
  grading.py      Answer extraction, hedge detection, error attribution.
  runner.py       Provider adapters, disk cache, mock models.
  report.py       Metrics, paired analysis, leaderboard export.
  cli.py          generate / run / verify-rates
web/index.html    Dashboard
tests/            70 tests
```

---

## Verifying the rates

`rates.py` is the file the whole benchmark rests on. A benchmark with wrong
ground truth is worse than no benchmark: it produces confident, citable, wrong
numbers.

Both tax years are currently marked `verified=False`, and `uktaxbench
verify-rates` exits non-zero until that changes. Flipping those flags means
opening each gov.uk page listed in `sources` and checking each figure by hand.

This is not a formality. While building this, one widely-syndicated tax site
listed the personal allowance as £13,000 (it is £12,570), and two student loan
calculators published 2024/25 thresholds under a 2026/27 heading. Secondary
sources are stale and mutually contradictory — which is both the motivation for
this benchmark and the reason its own figures need primary sourcing.

`figures_that_changed()` is guarded by a test asserting that *only* loan
thresholds differ between the two years. If a future year changes income tax as
well, the matched-pair design stops isolating a single variable, and that test
should fail loudly rather than let the analysis quietly become invalid.

---

## Limitations

**Not tax advice, and not a payroll engine.** The reference implementation
models the common PAYE case annually. It omits per-period NI (real Class 1 is
computed per pay period and is not cumulative, so uneven pay gives a different
answer), directors' annualised rules, employer NI, marriage allowance, blind
person's allowance, Gift Aid, and EIS/VCT/SEIS relief. Cases involving
irregular pay are excluded from generation for this reason.

**Attribution is a hypothesis, not a diagnosis.** When a model answers £0 to a
student loan question, `ignored_student_loan` matches trivially — as it would
for any model that answered £0 for any reason at all. Coincidental matches are
possible wherever a hypothesis value is also a common wrong answer. Treat
attribution as evidence of a pattern across many cases, never as an explanation
of any single one.

**Extraction is lenient by design.** A model writing "£1,234.00 per year"
instead of `1234` should not score zero — that measures instruction-following,
not tax knowledge. Parse method is recorded per case so format compliance can
be reported separately. Unparseable answers are scored as `None`, never as 0,
since 0 is sometimes correct.

**Hedge detection is a regex.** It catches conventional hedging language and
will miss uncertainty expressed in unusual ways. It is a floor on hedging, so
silent error rates should be read as an upper bound.

**Scottish figures need the closest scrutiny.** Scottish bands are stored as
taxable-income bounds converted from published gross thresholds, and savings
and dividend income use UK-wide rates and band widths even for Scottish
taxpayers. That is genuinely how it works, and it is the single most commonly
misstated part of the system — including, potentially, here.

---

## Licence

Code MIT. Rate figures are facts drawn from HMRC, gov.scot, and House of
Commons Library publications, cited in `rates.py`.
