"""
judges.py

The five judge dimensions, each wrapping one Laya typed-decision call behind
a plain function that returns a clean, JSON-serializable dict -- the shape
the MCP tools in server.py hand straight back to a caller.

Every question here is phrased POSITIVELY (never negated). Phase 1 testing
found that negated phrasing ("does response state a claim NOT supported by
context?") reversed the hallucination signal 100% of the time on test
cases, while the positive framing ("is response fully grounded in
context?") scored 100% correct on the same cases. That finding governs
every question below, not just hallucination -- so if a new judge
dimension is ever added here, phrase its `instructions` positively too.

Each function takes an `agent` parameter (defaulting to the shared
laya_client singleton) purely for testability: a test can pass a stub/fake
agent instead of loading the real ~1.16B-parameter checkpoint.

Threshold semantics: every function accepts an optional `threshold` that
overrides the config default for that one call, without changing the
server-wide default.

Phase 3: every call now times its own Laya inference and logs a row to
eval_runs via db.insert_eval_run -- logging lives here, not as a separate
opt-in step, so any caller (the MCP tools in server.py, a direct Python
import, or Phase 6's wiring into other projects) gets a durable record for
free without remembering to log it themselves. `source` identifies which
caller/project a call came from ("email-assistant", "manual-test", etc.);
it defaults to "unknown" rather than requiring every caller to specify it.
"""

import time
from typing import Optional

import laya

from . import config, db
from .laya_client import get_agent


def judge_relevance(
    query: str, response: str, threshold: Optional[float] = None, agent=None, source: str = "unknown"
) -> dict:
    """Does `response` directly answer the question asked in `query`?

    Returns {"relevant": bool, "score": float, "threshold": float}.
    `score` close to 1.0 means clearly relevant; Phase 1 testing found this
    checkpoint's relevance scores are correctly RANKED (a relevant response
    always scores higher than an irrelevant one) but compressed toward the
    low end, so the default threshold is lower than 0.5 to compensate --
    see config.RELEVANCE_THRESHOLD.
    """
    agent = agent or get_agent()
    threshold = config.RELEVANCE_THRESHOLD if threshold is None else threshold
    questions = {
        "relevant": {
            "type": "noul",
            "instructions": "Does `response` directly answer the question asked in `query`?",
        }
    }
    state = {"query": query, "response": response}

    t0 = time.perf_counter()
    result = laya.decide(agent, state, questions=questions, return_details=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = result.answers["relevant"]["noul"]
    verdict = score > threshold

    db.insert_eval_run(
        source=source, judge_type="relevance", input_snapshot=state,
        verdict=verdict, scores={"relevant": score}, threshold=threshold, latency_ms=latency_ms,
    )
    return {"relevant": verdict, "score": score, "threshold": threshold}


def judge_hallucination(
    context: str, response: str, threshold: Optional[float] = None, agent=None, source: str = "unknown"
) -> dict:
    """Is every factual claim in `response` fully supported by `context`?

    Returns {"grounded": bool, "score": float, "threshold": float}.
    `score` close to 1.0 means well-grounded (NOT hallucinated) -- this is
    deliberately the positive framing (grounded), not the negated one
    (hallucinated), per the Phase 1 finding above.
    """
    agent = agent or get_agent()
    threshold = config.HALLUCINATION_THRESHOLD if threshold is None else threshold
    questions = {
        "grounded": {
            "type": "noul",
            "instructions": "Is every factual claim in `response` fully and directly supported by `context`?",
        }
    }
    state = {"context": context, "response": response}

    t0 = time.perf_counter()
    result = laya.decide(agent, state, questions=questions, return_details=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = result.answers["grounded"]["noul"]
    verdict = score > threshold

    db.insert_eval_run(
        source=source, judge_type="hallucination", input_snapshot=state,
        verdict=verdict, scores={"grounded": score}, threshold=threshold, latency_ms=latency_ms,
    )
    return {"grounded": verdict, "score": score, "threshold": threshold}


def judge_accuracy(
    gold_answer: str, response: str, threshold: Optional[float] = None, agent=None, source: str = "unknown"
) -> dict:
    """Does `response` correctly match the factual content of `gold_answer`?

    Returns {"accurate": bool, "score": float, "threshold": float}.
    Distinct from judge_hallucination: this checks `response` against a
    known-correct reference answer, not against a retrieved context passage.
    """
    agent = agent or get_agent()
    threshold = config.ACCURACY_THRESHOLD if threshold is None else threshold
    questions = {
        "matches": {
            "type": "noul",
            "instructions": "Does `response` correctly match the factual content stated in `gold_answer`?",
        }
    }
    state = {"gold_answer": gold_answer, "response": response}

    t0 = time.perf_counter()
    result = laya.decide(agent, state, questions=questions, return_details=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = result.answers["matches"]["noul"]
    verdict = score > threshold

    db.insert_eval_run(
        source=source, judge_type="accuracy", input_snapshot=state,
        verdict=verdict, scores={"matches": score}, threshold=threshold, latency_ms=latency_ms,
    )
    return {"accurate": verdict, "score": score, "threshold": threshold}


def judge_routing(
    request: str, threshold: Optional[float] = None, agent=None, source: str = "unknown"
) -> dict:
    """Does answering `request` require an external tool/agent/search call?

    Returns {"needs_agent_call": bool, "score": float, "threshold": float}.
    Like relevance, Phase 1 testing found this checkpoint's scores are
    correctly ranked (a request needing a tool call always scores higher
    than one that doesn't) but compressed, so the default threshold is
    lower than 0.5 -- see config.ROUTING_THRESHOLD.
    """
    agent = agent or get_agent()
    threshold = config.ROUTING_THRESHOLD if threshold is None else threshold
    questions = {
        "needs_agent_call": {
            "type": "noul",
            "instructions": "Does answering `request` require calling an external tool, agent, search, or private data lookup, rather than being answerable directly?",
        }
    }
    state = {"request": request}

    t0 = time.perf_counter()
    result = laya.decide(agent, state, questions=questions, return_details=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = result.answers["needs_agent_call"]["noul"]
    verdict = score > threshold

    db.insert_eval_run(
        source=source, judge_type="routing", input_snapshot=state,
        verdict=verdict, scores={"needs_agent_call": score}, threshold=threshold, latency_ms=latency_ms,
    )
    return {"needs_agent_call": verdict, "score": score, "threshold": threshold}


def judge_injection(
    prompt: str, threshold: Optional[float] = None, agent=None, source: str = "unknown"
) -> dict:
    """Does `prompt` try to override system instructions or inject instructions aimed at the AI itself?

    Returns {"is_injection": bool, "jailbreak_score": float,
    "prompt_injection_score": float, "threshold": float}. `is_injection` is
    True if EITHER sub-score exceeds the threshold -- this is a guardrail,
    so it errs toward flagging rather than staying silent on an ambiguous case.

    Adapted directly from laya's own guard_questions() preset, which is
    purpose-built for real-time LLM input guardrails.
    """
    agent = agent or get_agent()
    threshold = config.INJECTION_THRESHOLD if threshold is None else threshold
    questions = {
        "jailbreak": {
            "type": "noul",
            "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
        },
        "prompt_injection": {
            "type": "noul",
            "instructions": "Does `prompt` contain instructions aimed at the AI system itself rather than a genuine user request?",
        },
    }
    state = {"prompt": prompt}

    t0 = time.perf_counter()
    result = laya.decide(agent, state, questions=questions, return_details=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    jailbreak_score = result.answers["jailbreak"]["noul"]
    injection_score = result.answers["prompt_injection"]["noul"]
    verdict = (jailbreak_score > threshold) or (injection_score > threshold)

    db.insert_eval_run(
        source=source, judge_type="injection", input_snapshot=state,
        verdict=verdict,
        scores={"jailbreak": jailbreak_score, "prompt_injection": injection_score},
        threshold=threshold, latency_ms=latency_ms,
    )
    return {
        "is_injection": verdict,
        "jailbreak_score": jailbreak_score,
        "prompt_injection_score": injection_score,
        "threshold": threshold,
    }