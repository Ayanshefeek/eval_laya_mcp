# eval-judge-mcp

Judges agent/RAG outputs — relevance, hallucination, accuracy, routing, and
prompt-injection — using [Laya](https://laya-ai.com), a small (~1.16B param)
open-weights typed-decision model, instead of a full LLM-as-judge call.
Every judgment is logged to a local SQLite database and can be aggregated
into pass rates, average scores, and latency via a built-in reporting tool.

## Two ways to use this — pick based on your setup

This project is **not only an MCP server**. The judging logic lives in a
plain, importable Python module (`eval_judge_mcp/judges.py`) that has no
dependency on the MCP protocol at all — the MCP server is a thin adapter
layered on top of it. Use whichever path fits how you're building:

| You're building... | Use this |
|---|---|
| An agent in Claude Desktop, Cursor, or any other MCP-capable client/framework | **Run it as an MCP server** (below) — the agent calls `judge_hallucination`, `judge_relevance`, etc. as tools, zero code needed on your side. |
| A plain Python app — a RAG pipeline, a FastAPI backend, a notebook, anything without an MCP client in the loop | **Import the functions directly** — `pip install eval-judge-mcp`, then `from eval_judge_mcp.judges import judge_hallucination`. No server, no protocol overhead, same calibrated thresholds and eval logging either way. |

You can also do both in the same project — e.g. call the functions directly
inside your RAG pipeline's code, while also exposing the MCP server for an
agent that orchestrates that pipeline.

### Path A: as an MCP server

Add it to your MCP client config (e.g. Claude Desktop's
`claude_desktop_config.json`, Cursor's `mcp.json`):

```json
{
  "mcpServers": {
    "eval-judge": {
      "command": "uvx",
      "args": ["eval-judge-mcp"]
    }
  }
}
```

Your agent can then call `judge_relevance`, `judge_hallucination`,
`judge_accuracy`, `judge_routing`, `judge_injection`, and `get_eval_summary`
as tools mid-conversation.

Running it manually (e.g. for local testing with the MCP Inspector):

```bash
mcp dev eval_judge_mcp/server.py:mcp
# or
python -m eval_judge_mcp.server
```

### Path B: as a plain Python library

```bash
pip install eval-judge-mcp
```

```python
from eval_judge_mcp.judges import judge_hallucination, judge_relevance

result = judge_hallucination(
    context="The meeting is rescheduled to 3pm Thursday.",
    response="The meeting is now at 3pm on Thursday.",
    source="my-rag-app",  # tags this call in the eval log
)
# {"grounded": True, "score": 0.91, "threshold": 0.5}

if not result["grounded"]:
    # flag it, log it, ask for regeneration, whatever fits your pipeline
    ...
```

Every call — whether it comes through the MCP server or a direct import —
writes to the same eval log by default, so `get_eval_summary()` reports on
both uniformly. See **Opting out of logging** below if you don't want that.

## Why not just call Laya directly?

You can — but this project exists because getting good results out of Laya
for these five judgments isn't just "call `laya.decide()`":

- **Question phrasing matters more than expected.** A negated question
  ("does `response` state a claim NOT supported by `context`?") reversed the
  hallucination signal 100% of the time in testing; the positive framing
  ("is `response` fully grounded in `context`?") scored 100% correct on the
  same cases. Every question here is deliberately phrased positively.
- **A flat 0.5 threshold misclassifies two of the five dimensions.**
  Relevance and routing scores came back correctly *ranked* (good cases
  always score higher than bad ones) but *compressed* toward the low end —
  so their default thresholds are tuned lower (0.40 and 0.30) to compensate.
  See `config.py` for the full rationale and how to override any threshold
  via environment variables.
- **No logging or reporting comes for free from the library.** Every judge
  call here is automatically logged (timestamp, source, scores, verdict,
  latency) to SQLite, and `get_eval_summary()` aggregates that into pass
  rates and averages — grouped by judge type, filterable by `source` and
  `since_days`. Calling Laya directly gets you a score and nothing else.

## The five judge dimensions

| Function | Question it answers | Returns |
|---|---|---|
| `judge_relevance(query, response)` | Does `response` directly answer `query`? | `{relevant, score, threshold}` |
| `judge_hallucination(context, response)` | Is every claim in `response` supported by `context`? | `{grounded, score, threshold}` |
| `judge_accuracy(gold_answer, response)` | Does `response` match the known-correct `gold_answer`? | `{accurate, score, threshold}` |
| `judge_routing(request)` | Does answering `request` need a tool/agent/search call? | `{needs_agent_call, score, threshold}` |
| `judge_injection(prompt)` | Does `prompt` attempt a jailbreak or prompt injection? | `{is_injection, jailbreak_score, prompt_injection_score, threshold}` |

`judge_hallucination` vs `judge_accuracy`: hallucination checks a response
against a *retrieved passage* (is it grounded in what was fetched?);
accuracy checks a response against a *known-correct reference answer*,
regardless of what context produced it. A response can be grounded in
context but still wrong if the context itself was incomplete — these two
dimensions catch different failure modes.

`judge_injection` is aimed at RAG/agentic pipelines specifically: it's
worth running not just on user input, but on retrieved documents before
they're inserted into a prompt, to catch indirect prompt injection from
poisoned sources.

Every function accepts an optional `threshold` (overrides the default for
that one call), `source` (tags which project/caller logged it, defaults to
`"unknown"`), and `log` (overrides whether this one call gets persisted to
the eval database — see **Opting out of logging** below).

## Reporting

```python
from eval_judge_mcp.db import get_eval_summary

get_eval_summary(source="my-rag-app", since_days=7)
```

```json
{
  "filters": {"source": "my-rag-app", "judge_type": null, "since_days": 7},
  "total_runs": 42,
  "by_judge_type": {
    "hallucination": {
      "count": 20, "pass_rate": 0.85,
      "avg_score": 0.78, "avg_latency_ms": 810.2
    }
  }
}
```

Available as `get_eval_summary` through the MCP server too, or call
`eval_judge_mcp.db.get_recent_runs(...)` directly for the raw per-call rows.

## Opting out of logging

By default every judge call writes a row to SQLite — that's what makes
`get_eval_summary()` possible. If you don't want that (a read-only
filesystem, or not wanting query/response content persisted to disk at
all), there are two levels of control, same pattern as `threshold`:

**Per-call**, pass `log=False` on the calls you don't want persisted:

```python
judge_hallucination(context, response, log=False)  # never written to disk
```

**Globally**, turn it off entirely with an environment variable before your
app/server starts:

```bash
export EVAL_JUDGE_LOGGING_ENABLED=false
```

With logging disabled at the global level, the MCP server also skips
creating the SQLite file at startup — not just skipping writes to it — so
nothing touches disk unless you explicitly override a specific call with
`log=True`. A per-call `log=True`/`log=False` always wins over the global
setting; `log` left unset (the default) follows whatever
`EVAL_JUDGE_LOGGING_ENABLED` says (on, by default).

## Configuration

Every tunable value is read from an environment variable with a sensible
default — see `config.py`. Notably:

- `LAYA_MODEL_ID`, `LAYA_SUBFOLDER`, `LAYA_DEVICE` — which checkpoint to
  load and where (`LAYA_DEVICE` unset lets Laya auto-detect CPU/CUDA)
- `EVAL_JUDGE_DB_PATH` — where the SQLite eval log lives (default
  `eval_runs.db` in the working directory)
- `EVAL_JUDGE_LOGGING_ENABLED` — global on/off switch for eval logging
  (default `true`) — see **Opting out of logging** above
- `EVAL_JUDGE_RELEVANCE_THRESHOLD`, `EVAL_JUDGE_HALLUCINATION_THRESHOLD`,
  `EVAL_JUDGE_ACCURACY_THRESHOLD`, `EVAL_JUDGE_ROUTING_THRESHOLD`,
  `EVAL_JUDGE_INJECTION_THRESHOLD` — per-dimension pass/fail cutoffs

Threshold defaults come from a small calibration pass (5 hand-picked pairs
per dimension) — a reasonable starting point, not a validated ground truth.
Override them via env vars as you gather real labeled data from your own
traffic, rather than editing the defaults in `config.py` directly.

## Requirements and first-run cost

This depends on `laya`, which pulls in `torch` and downloads the
`convaiinnovations/laya` checkpoint (~2-3GB) from Hugging Face on first use.
Plan for:

- A few GB of free disk for the checkpoint
- A one-time download on first run (subsequent runs load from cache)
- CPU inference works but is slower (~500-900ms per call in testing); set
  `LAYA_DEVICE=cuda` if a GPU is available for meaningfully lower latency

