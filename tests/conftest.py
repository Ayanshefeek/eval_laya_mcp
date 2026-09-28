"""
conftest.py

The real `laya` package pulls in torch and a multi-GB model checkpoint, which
is exactly what judges.py's `agent` parameter (see its module docstring) is
designed to let tests avoid -- but judges.py also does `import laya` at
module level and calls `laya.decide(...)` directly, so importing judges.py
at all requires *some* `laya` module to exist, real or fake.

So before any test imports eval_judge_mcp.judges (or server, which imports
judges transitively), we inject a fake `laya` module into sys.modules with a
controllable `decide()`. This keeps the test suite fast and independent of
network access / GPU / the real checkpoint, while still exercising the real
threshold/verdict/logging logic in judges.py against a scriptable fake.

Each test controls what the fake returns via the `fake_laya` fixture's
`.set_answers(...)`, rather than a single hardcoded response, since different
tests need different scores to hit different branches (pass/fail thresholds,
the injection OR logic).
"""

import sys
import types
from pathlib import Path

import pytest

# Make the project root (parent of eval_judge_mcp/) importable, mirroring the
# sys.path shim server.py uses for standalone loading.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeDecisionResult:
    """Stands in for laya's DecisionResult: judges.py only reads .answers,
    shaped as {question_name: {question_type: value}}."""

    def __init__(self, answers):
        self.answers = answers


class _FakeLayaModule(types.ModuleType):
    """Minimal stand-in for the `laya` package. Only implements what
    judges.py actually calls: `laya.decide(agent, state, questions=...,
    return_details=True)` -> object with `.answers`.

    `load()` is stubbed too so laya_client.py stays importable, though the
    judge tests never call it -- they inject a fake agent directly.
    """

    def __init__(self):
        super().__init__("laya")
        self._next_answers = None

    def set_answers(self, answers):
        """Queue the `.answers` dict the next `decide()` call should return.
        `answers` is {question_name: {question_type: score_or_value}}."""
        self._next_answers = answers

    def decide(self, agent, state, questions=None, return_details=False):
        if self._next_answers is None:
            raise AssertionError(
                "fake_laya.set_answers(...) must be called before the judge "
                "function under test invokes laya.decide()"
            )
        return _FakeDecisionResult(self._next_answers)

    def load(self, model_id, subfolder=None, device=None):
        return object()  # never actually used by the judge-function tests


# Installed at collection time, not inside a fixture: test_judges.py does
# `import eval_judge_mcp.judges` at MODULE level (so it can reference
# judges.judge_relevance etc. directly), and that import chain hits
# judges.py's own `import laya` before any per-test fixture gets a chance to
# run. Registering the fake here, as soon as conftest.py itself is imported
# (pytest always imports conftest.py before collecting test modules in the
# same directory), makes sure `import laya` already resolves to the fake by
# the time test_judges.py is collected.
sys.modules.setdefault("laya", _FakeLayaModule())


@pytest.fixture
def fake_laya(monkeypatch):
    """Hand back the fake `laya` module installed above, reset for this one
    test so a leftover .set_answers(...) from a previous test can't leak in."""
    import laya as fake  # resolves to the fake module registered above

    fake._next_answers = None
    monkeypatch.setattr(fake, "_next_answers", None, raising=False)

    # judges.py's module-level `import laya` bound its own `laya` name at
    # import time -- re-point that binding at this same fake object too
    # (it's a no-op once the fake is already installed via sys.modules
    # before judges.py's first import, but keeps this fixture correct even
    # if import order ever changes).
    import eval_judge_mcp.judges as judges_module

    monkeypatch.setattr(judges_module, "laya", fake, raising=False)
    return fake


@pytest.fixture
def fresh_db(monkeypatch, tmp_path):
    """Point the eval DB at a fresh temp file for one test, and force
    db.init_db() to re-run against it.

    db._initialized is a module-level flag set True the first time ANY test
    (or the real server) creates the table -- it doesn't track *which* path
    was initialized. Without resetting it here, a test using a second temp
    path would skip schema creation and fail with "no such table: eval_runs"
    the moment the previous test in the same process already initialized a
    different path.
    """
    import eval_judge_mcp.config as config
    import eval_judge_mcp.db as db

    db_path = tmp_path / "eval_runs_test.db"
    monkeypatch.setattr(config, "EVAL_DB_PATH", str(db_path))
    monkeypatch.setattr(db, "_initialized", False)
    db.init_db()
    return db_path