"""
server.py

The MCP server itself: exposes the five judge functions from judges.py as
MCP tools, so any MCP client (Claude Desktop, Cursor, etc.) can call them.

The official `mcp` SDK renamed its high-level server class between major
versions: `mcp.server.fastmcp.FastMCP` in 1.x, `mcp.server.mcpserver.MCPServer`
in 2.x (2.x is current on PyPI as of this writing -- `pip install mcp` today
gets 2.x by default). Verified directly, not assumed: this project hit both
failure modes in testing -- importing FastMCP against a real 2.x install
raises `ModuleNotFoundError: No module named 'mcp.server.fastmcp'` with an
explicit message pointing at this rename, and importing MCPServer against a
1.x install fails the same way in reverse. So this imports whichever one
the installed `mcp` actually provides, tried newest-API-first, rather than
assuming a version. `name` and `instructions` are the constructor kwargs
confirmed common to both classes' signatures; anything version-specific
(2.x's extra `title`/`description` kwargs, for example) is deliberately not
used here, to keep one code path working across both.

This module intentionally contains no judging logic itself -- it is a thin
adapter layer. All the actual decision-making (and, as of Phase 3, eval
logging) lives in judges.py, which can be tested and used independently of
MCP (e.g. from a plain Python script, or wired directly into an agent
pipeline without going through the MCP protocol at all).

Import note: `mcp dev` (the MCP Inspector CLI) loads this file directly via
importlib, outside of the eval_judge_mcp package context, so a relative
import ("from . import judges") fails with "attempted relative import with
no known parent package". Running via `python -m eval_judge_mcp.server`
doesn't hit this, since -m sets up the package properly -- but the server
needs to work under both invocation styles. The sys.path shim below makes
the package's parent directory importable, so an absolute import
("eval_judge_mcp.judges") works regardless of how this file was loaded.
"""

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Loaded as a standalone file (e.g. by `mcp dev`) -- add the project
    # root so `eval_judge_mcp` can be imported absolutely.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from mcp.server.mcpserver import MCPServer as _ServerClass  # mcp 2.x (current)
except ModuleNotFoundError:
    from mcp.server.fastmcp import FastMCP as _ServerClass  # mcp 1.x (legacy)

from eval_judge_mcp import config, db, judges

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if config.EVAL_LOGGING_ENABLED:
    db.init_db()  # create eval_runs.db / the eval_runs table if this is the first run
# When logging is disabled (EVAL_JUDGE_LOGGING_ENABLED=false), skip eager
# init entirely -- otherwise the db file would get created at startup even
# for someone who explicitly opted out, defeating the point (e.g. on a
# read-only filesystem). get_eval_summary/get_recent_runs still lazily
# create the table on demand if called directly with log=True on some
# calls, which is fine -- that's an explicit request to persist something.

mcp = _ServerClass(
    name="eval-judge-mcp",
    instructions="Judges agent outputs (relevance, hallucination, accuracy, routing, prompt injection) using Laya, a small open-weights typed-decision model, instead of a full LLM-as-judge.",
)


@mcp.tool()
def judge_relevance(
    query: str, response: str, threshold: float | None = None,
    source: str = "unknown", log: bool | None = None,
) -> dict:
    """Judge whether `response` directly answers the question asked in `query`.

    `source` identifies which project/caller this call came from (e.g.
    "email-assistant") -- it's recorded in the eval log, defaulting to
    "unknown" if not given. `log` overrides the server's default logging
    setting for this one call: omit it to use the configured default
    (EVAL_JUDGE_LOGGING_ENABLED, on by default), pass False to skip logging
    this call, or True to force logging even if the default is off.

    Returns a dict with `relevant` (bool), `score` (0-1 probability), and
    `threshold` (the cutoff actually used for this call).
    """
    return judges.judge_relevance(query, response, threshold, source=source, log=log)


@mcp.tool()
def judge_hallucination(
    context: str, response: str, threshold: float | None = None,
    source: str = "unknown", log: bool | None = None,
) -> dict:
    """Judge whether every factual claim in `response` is supported by `context`.

    See judge_relevance's docstring for what `log` does.

    Returns a dict with `grounded` (bool, True = NOT hallucinated),
    `score` (0-1 probability of being grounded), and `threshold`.
    """
    return judges.judge_hallucination(context, response, threshold, source=source, log=log)


@mcp.tool()
def judge_accuracy(
    gold_answer: str, response: str, threshold: float | None = None,
    source: str = "unknown", log: bool | None = None,
) -> dict:
    """Judge whether `response` correctly matches the factual content of `gold_answer`.

    See judge_relevance's docstring for what `log` does.

    Returns a dict with `accurate` (bool), `score` (0-1 probability), and
    `threshold`.
    """
    return judges.judge_accuracy(gold_answer, response, threshold, source=source, log=log)


@mcp.tool()
def judge_routing(
    request: str, threshold: float | None = None,
    source: str = "unknown", log: bool | None = None,
) -> dict:
    """Judge whether answering `request` requires an external tool/agent/search call.

    See judge_relevance's docstring for what `log` does.

    Returns a dict with `needs_agent_call` (bool), `score` (0-1
    probability), and `threshold`.
    """
    return judges.judge_routing(request, threshold, source=source, log=log)


@mcp.tool()
def judge_injection(
    prompt: str, threshold: float | None = None,
    source: str = "unknown", log: bool | None = None,
) -> dict:
    """Judge whether `prompt` attempts a jailbreak or prompt injection.

    See judge_relevance's docstring for what `log` does.

    Returns a dict with `is_injection` (bool), `jailbreak_score` (0-1),
    `prompt_injection_score` (0-1), and `threshold`.
    """
    return judges.judge_injection(prompt, threshold, source=source, log=log)


@mcp.tool()
def get_eval_summary(source: str | None = None, judge_type: str | None = None, since_days: int | None = None) -> dict:
    """Summarize logged eval_runs: count, pass rate, average score, and
    average latency, grouped by judge dimension.

    Optionally filter by `source` (which project/caller logged the runs),
    `judge_type` (one of relevance/hallucination/accuracy/routing/injection),
    and `since_days` (only runs from the last N days; omit for all time).
    """
    return db.get_eval_summary(source=source, judge_type=judge_type, since_days=since_days)


def main() -> None:
    """Entry point for the `eval-judge-mcp` console script (what `uvx
    eval-judge-mcp` and a `pip`-installed console script both invoke) --
    see pyproject.toml's [project.scripts]."""
    mcp.run()


if __name__ == "__main__":
    main()