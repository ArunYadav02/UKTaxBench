"""
Running cases against models.

Two design decisions worth keeping:

1. Everything is cached on disk, keyed by a hash of (model, prompt, params).
   Benchmark development means re-running constantly while you fix the
   harness. Without a cache you pay for the same tokens repeatedly, and the
   bill is what quietly kills student benchmark projects.

2. A `mock` provider ships in the box. The whole pipeline — generate, run,
   grade, report, visualise — works with no API key and no spend. Build and
   debug the harness against the mock, then swap in a real model once the
   plumbing is proven. The mock deliberately makes *realistic* mistakes
   (stale thresholds, forgotten loans) so the error-attribution code has
   something to find.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

D = Decimal

CACHE_DIR = Path(os.environ.get("UKTAXBENCH_CACHE", ".cache/responses"))


SYSTEM_PROMPT = (
    "You are answering questions about UK personal taxation. "
    "Work through the calculation, then give your final answer on the last "
    "line in the exact form <answer>NUMBER</answer> with no currency symbol, "
    "no commas, and no units. If you are uncertain, say so explicitly."
)


@dataclass
class Response:
    model: str
    case_id: str
    text: str
    cached: bool
    latency_s: float
    error: str | None = None


def _key(model: str, prompt: str, temperature: float) -> str:
    blob = json.dumps(
        {"m": model, "p": prompt, "t": temperature, "s": SYSTEM_PROMPT},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _read_cache(key: str) -> str | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())["text"]
        except Exception:
            return None
    return None


def _write_cache(key: str, text: str) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"text": text}))


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

Provider = Callable[[str, str, float], str]


def anthropic_provider(model: str, prompt: str, temperature: float) -> str:
    import anthropic  # imported lazily so the package works without it

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def openai_provider(model: str, prompt: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def mock_provider(model: str, prompt: str, temperature: float) -> str:
    """A fake model that is right most of the time and wrong in realistic ways.

    Its errors are drawn from the same failure modes the grader looks for, so
    you can verify that error attribution actually fires before spending money
    on real inference. Seeded by prompt hash, so it is deterministic.
    """
    from .generator import CASES_BY_PROMPT  # populated when cases are built

    seed = int(hashlib.sha256((model + prompt).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    case = CASES_BY_PROMPT.get(prompt)
    if case is None:
        return "I could not determine the figure. <answer>0</answer>"

    truth = D(str(case["answer"]))
    skill = {"mock-good": 0.86, "mock-mid": 0.62, "mock-poor": 0.38}.get(model, 0.62)

    if rng.random() < skill:
        val = truth
        hedge = ""
    else:
        mode = rng.choice(["stale", "drop_loan", "round", "way_off"])
        if mode == "stale":
            from .grading import _rehydrate, build_hypotheses
            try:
                alt = build_hypotheses(case["answer_field"])["stale_year"](
                    _rehydrate(case["taxpayer"])
                )
            except Exception:
                alt = None
            val = alt if alt is not None else truth * D("1.03")
        elif mode == "drop_loan":
            from .grading import _rehydrate, build_hypotheses
            try:
                alt = build_hypotheses(case["answer_field"])["ignored_student_loan"](
                    _rehydrate(case["taxpayer"])
                )
            except Exception:
                alt = None
            val = alt if alt is not None else truth * D("0.94")
        elif mode == "round":
            val = (truth / D("100")).to_integral_value() * D("100")
        else:
            val = truth * D(str(round(rng.uniform(0.7, 1.3), 3)))
        hedge = "This is approximate; please verify with HMRC. " if rng.random() < 0.35 else ""

    return (
        f"{hedge}Working through the bands and thresholds for the stated tax year, "
        f"the figure comes out as follows.\n<answer>{val:.2f}</answer>"
    )


PROVIDERS: dict[str, Provider] = {
    "anthropic": anthropic_provider,
    "openai": openai_provider,
    "mock": mock_provider,
}


def provider_for(model: str) -> Provider:
    if model.startswith("mock"):
        return PROVIDERS["mock"]
    if model.startswith("claude"):
        return PROVIDERS["anthropic"]
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return PROVIDERS["openai"]
    raise ValueError(
        f"No provider mapped for model {model!r}. Add one in runner.PROVIDERS."
    )


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def run_case(
    model: str,
    case: dict[str, Any],
    temperature: float = 0.0,
    retries: int = 3,
    use_cache: bool = True,
) -> Response:
    prompt = case["prompt"]
    key = _key(model, prompt, temperature)

    if use_cache:
        hit = _read_cache(key)
        if hit is not None:
            return Response(model, case["id"], hit, True, 0.0)

    fn = provider_for(model)
    start = time.time()
    last_err: str | None = None
    for attempt in range(retries):
        try:
            text = fn(model, prompt, temperature)
            if use_cache:
                _write_cache(key, text)
            return Response(model, case["id"], text, False, time.time() - start)
        except Exception as exc:  # noqa: BLE001 - we want the message, not the type
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 8))
    return Response(model, case["id"], "", False, time.time() - start, error=last_err)


def run_all(
    model: str,
    cases: Iterable[dict[str, Any]],
    temperature: float = 0.0,
    use_cache: bool = True,
    progress: bool = True,
) -> list[Response]:
    cases = list(cases)
    out: list[Response] = []
    for i, c in enumerate(cases, 1):
        out.append(run_case(model, c, temperature, use_cache=use_cache))
        if progress and (i % 25 == 0 or i == len(cases)):
            hits = sum(r.cached for r in out)
            print(f"  {model}: {i}/{len(cases)}  (cache hits {hits})", flush=True)
    return out
