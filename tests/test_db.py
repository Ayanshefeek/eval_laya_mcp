"""
test_db.py

Unit tests for eval_judge_mcp/db.py: insert/read round-tripping and, most
importantly, the get_eval_summary aggregation math -- hand-computed against
known inserted rows, since that's the part with real logic (not just a
passthrough SQL SELECT) and the part get_eval_summary's own docstring is
most careful to explain (per-value score averaging vs. per-row pass rate).
"""

import sqlite3

import pytest

import eval_judge_mcp.db as db


# --- insert_eval_run / get_recent_runs --------------------------------------

def test_insert_and_read_round_trip(fresh_db):
    row_id = db.insert_eval_run(
        source="coding-assistant",
        judge_type="relevance",
        input_snapshot={"query": "q", "response": "r"},
        verdict=True,
        scores={"relevant": 0.81},
        threshold=0.40,
        latency_ms=123.4,
    )
    assert isinstance(row_id, int)

    rows = db.get_recent_runs(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == row_id
    assert row["source"] == "coding-assistant"
    assert row["judge_type"] == "relevance"
    assert row["verdict"] is True
    assert row["scores"] == {"relevant": 0.81}
    assert row["input_snapshot"] == {"query": "q", "response": "r"}
    assert row["threshold"] == 0.40
    assert row["latency_ms"] == 123.4


def test_get_recent_runs_newest_first_and_limit(fresh_db):
    for i in range(5):
        db.insert_eval_run(
            source="s", judge_type="relevance", input_snapshot={"i": i},
            verdict=True, scores={"relevant": 0.5}, threshold=0.4, latency_ms=1.0,
        )
    rows = db.get_recent_runs(limit=2)
    assert len(rows) == 2
    assert rows[0]["input_snapshot"]["i"] == 4  # most recent inserted last -> first out
    assert rows[1]["input_snapshot"]["i"] == 3


def test_get_recent_runs_filters_by_source_and_judge_type(fresh_db):
    db.insert_eval_run(source="email-assistant", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=1.0)
    db.insert_eval_run(source="coding-assistant", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=1.0)
    db.insert_eval_run(source="email-assistant", judge_type="accuracy", input_snapshot={},
                        verdict=False, scores={"matches": 0.1}, threshold=0.5, latency_ms=1.0)

    by_source = db.get_recent_runs(limit=10, source="email-assistant")
    assert len(by_source) == 2
    assert all(r["source"] == "email-assistant" for r in by_source)

    by_judge_type = db.get_recent_runs(limit=10, judge_type="relevance")
    assert len(by_judge_type) == 2
    assert all(r["judge_type"] == "relevance" for r in by_judge_type)

    both = db.get_recent_runs(limit=10, source="email-assistant", judge_type="accuracy")
    assert len(both) == 1
    assert both[0]["judge_type"] == "accuracy"


# --- get_eval_summary: aggregation math --------------------------------------

def test_summary_empty_db(fresh_db):
    summary = db.get_eval_summary()
    assert summary["total_runs"] == 0
    assert summary["by_judge_type"] == {}
    assert summary["filters"] == {"source": None, "judge_type": None, "since_days": None}


def test_summary_pass_rate_and_avg_latency_single_score_judge(fresh_db):
    # 3 relevance runs: 2 pass, 1 fail -> pass_rate 2/3
    db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=100.0)
    db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 0.7}, threshold=0.4, latency_ms=200.0)
    db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                        verdict=False, scores={"relevant": 0.1}, threshold=0.4, latency_ms=300.0)

    summary = db.get_eval_summary()
    stats = summary["by_judge_type"]["relevance"]
    assert stats["count"] == 3
    assert stats["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert stats["avg_score"] == pytest.approx((0.9 + 0.7 + 0.1) / 3, abs=1e-4)
    assert stats["avg_latency_ms"] == pytest.approx(200.0, abs=1e-4)


def test_summary_avg_score_counts_each_injection_subscore_separately(fresh_db):
    # judge_injection logs TWO scores per row (jailbreak, prompt_injection).
    # avg_score must average all 4 individual values here, not the 2 row-level
    # verdicts -- per get_eval_summary's own docstring.
    db.insert_eval_run(
        source="s", judge_type="injection", input_snapshot={}, verdict=True,
        scores={"jailbreak": 0.9, "prompt_injection": 0.1}, threshold=0.5, latency_ms=50.0,
    )
    db.insert_eval_run(
        source="s", judge_type="injection", input_snapshot={}, verdict=False,
        scores={"jailbreak": 0.2, "prompt_injection": 0.3}, threshold=0.5, latency_ms=150.0,
    )

    stats = db.get_eval_summary()["by_judge_type"]["injection"]
    assert stats["count"] == 2  # row count, not score count
    assert stats["pass_rate"] == pytest.approx(1 / 2, abs=1e-4)  # from verdict, unaffected by score count
    expected_avg_score = (0.9 + 0.1 + 0.2 + 0.3) / 4
    assert stats["avg_score"] == pytest.approx(expected_avg_score, abs=1e-4)
    assert stats["avg_latency_ms"] == pytest.approx(100.0, abs=1e-4)


def test_summary_groups_by_judge_type_independently(fresh_db):
    db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 1.0}, threshold=0.4, latency_ms=10.0)
    db.insert_eval_run(source="s", judge_type="accuracy", input_snapshot={},
                        verdict=False, scores={"matches": 0.0}, threshold=0.5, latency_ms=20.0)

    summary = db.get_eval_summary()
    assert summary["total_runs"] == 2
    assert set(summary["by_judge_type"].keys()) == {"relevance", "accuracy"}
    assert summary["by_judge_type"]["relevance"]["pass_rate"] == 1.0
    assert summary["by_judge_type"]["accuracy"]["pass_rate"] == 0.0


def test_summary_filters_by_source(fresh_db):
    db.insert_eval_run(source="email-assistant", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=10.0)
    db.insert_eval_run(source="coding-assistant", judge_type="relevance", input_snapshot={},
                        verdict=False, scores={"relevant": 0.1}, threshold=0.4, latency_ms=10.0)

    summary = db.get_eval_summary(source="email-assistant")
    assert summary["total_runs"] == 1
    assert summary["by_judge_type"]["relevance"]["pass_rate"] == 1.0
    assert summary["filters"]["source"] == "email-assistant"


def test_summary_filters_by_judge_type(fresh_db):
    db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                        verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=10.0)
    db.insert_eval_run(source="s", judge_type="accuracy", input_snapshot={},
                        verdict=True, scores={"matches": 0.9}, threshold=0.5, latency_ms=10.0)

    summary = db.get_eval_summary(judge_type="accuracy")
    assert summary["total_runs"] == 1
    assert list(summary["by_judge_type"].keys()) == ["accuracy"]


def test_summary_since_days_excludes_older_rows(fresh_db):
    recent_id = db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                                    verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=10.0)
    old_id = db.insert_eval_run(source="s", judge_type="relevance", input_snapshot={},
                                 verdict=True, scores={"relevant": 0.9}, threshold=0.4, latency_ms=10.0)

    # insert_eval_run always stamps "now" -- backdate the second row directly
    # to simulate a run from 30 days ago, since there's no public API for it.
    conn = sqlite3.connect(fresh_db)
    conn.execute(
        "UPDATE eval_runs SET timestamp = datetime('now', '-30 days') WHERE id = ?",
        (old_id,),
    )
    conn.commit()
    conn.close()

    summary = db.get_eval_summary(since_days=7)
    assert summary["total_runs"] == 1
    assert summary["by_judge_type"]["relevance"]["count"] == 1

    unfiltered = db.get_eval_summary()
    assert unfiltered["total_runs"] == 2