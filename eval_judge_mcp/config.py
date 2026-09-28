"""
config.py

Centralizes every tunable value for the judge tools: which Laya checkpoint to
load, and the pass/fail threshold each judge dimension uses. Everything here
is read from an environment variable with a sensible default, never hardcoded
inline in judges.py or server.py, so deployment-time tuning never requires
editing code.

Threshold defaults come from Phase 1 calibration testing (5 hand-picked
pairs per dimension, see project notes), not guesswork:
  - injection, hallucination (grounded), accuracy (matches): the model's
    default 0.5 decision boundary correctly separated good/bad cases in
    testing, once questions were phrased POSITIVELY (never negated -- a
    negated proposition reversed the hallucination signal entirely).
  - relevance, routing (needs_agent_call): the model's scores were
    consistently ranked correctly (good > bad in every tested pair) but
    compressed toward the low end, so a flat 0.5 cutoff misclassified valid
    "good" cases. Lower thresholds compensate for that compression.
These defaults are a reasonable starting point from a SMALL sample (5 pairs
each) -- treat them as provisional until validated against a larger labeled
set from real agent traffic, and override via environment variables as that
evidence comes in, rather than editing this file's defaults directly.
"""

import os

# --- Laya checkpoint ---
LAYA_MODEL_ID = os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya")
LAYA_SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", "typed-decisions")
LAYA_DEVICE = os.environ.get("LAYA_DEVICE")  # None lets laya auto-detect (cpu/cuda)

# --- Eval logging ---
EVAL_DB_PATH = os.environ.get("EVAL_JUDGE_DB_PATH", "eval_runs.db")

# --- Per-dimension thresholds ---
# A score strictly greater than the threshold counts as a positive verdict
# for that dimension's semantic meaning (see judges.py docstrings for what
# "positive" means per tool).
RELEVANCE_THRESHOLD = float(os.environ.get("EVAL_JUDGE_RELEVANCE_THRESHOLD", "0.40"))
HALLUCINATION_THRESHOLD = float(os.environ.get("EVAL_JUDGE_HALLUCINATION_THRESHOLD", "0.50"))
ACCURACY_THRESHOLD = float(os.environ.get("EVAL_JUDGE_ACCURACY_THRESHOLD", "0.50"))
ROUTING_THRESHOLD = float(os.environ.get("EVAL_JUDGE_ROUTING_THRESHOLD", "0.30"))
INJECTION_THRESHOLD = float(os.environ.get("EVAL_JUDGE_INJECTION_THRESHOLD", "0.50"))