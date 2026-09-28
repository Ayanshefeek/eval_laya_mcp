"""
test_setup.py

Phase 1, steps 1-2 of the eval-judge MCP project: confirm the `typed-decisions`
Laya checkpoint loads correctly on this machine and can answer a basic typed
question end to end, before building anything on top of it.

We load `typed-decisions` directly rather than going through laya.Router,
since this project only ever needs the typed-decision checkpoint -- no
language routing or general-purpose text tasks are involved. That keeps the
download to one ~2-3GB checkpoint instead of up to ~7GB across all three.

Each question needs an `instructions` field (the actual natural-language
question posed to the model against `state`) -- a bare options list is not
enough. `choice` and `score` describe each option in words via `criteria`
rather than just naming it, which is what actually steers the model's
judgment. Confirmed against laya's own triage_questions/email_questions
presets, since the initial schema guess (options without instructions) failed
with: ValueError: question 'department': no 'instructions'.
"""

import logging
import time

import laya

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_judge_model() -> "laya.agent.Agent":
    """Load and return the typed-decisions checkpoint, timing the cold load."""
    logger.info("Loading Laya typed-decisions checkpoint (first run downloads it)...")
    t0 = time.time()
    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
    logger.info("Loaded in %.1fs", time.time() - t0)
    return agent


def smoke_test(agent) -> None:
    """Run one basic typed decision to confirm the model actually answers correctly."""
    state = "I was charged twice for my subscription. Please refund the duplicate charge."

    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this support ticket in `state`?",
            "criteria": {
                "billing": "invoices, payments, refunds, duplicate charges",
                "technical": "bugs, outages, integration problems",
                "sales": "pricing questions, demos, new purchases",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request in `state`?",
            "criteria": ["low - no time pressure", "medium - needs attention soon", "critical - blocking issue"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for money back in `state`?",
        },
    }

    logger.info("Running smoke test decision...")
    t0 = time.time()
    result = laya.decide(agent, state, questions=questions, return_details=True)
    elapsed = time.time() - t0

    logger.info("Decision took %.2fs", elapsed)
    print("\n--- Smoke test result ---")
    print(result)


if __name__ == "__main__":
    agent = load_judge_model()
    smoke_test(agent)