# -*- coding: utf-8 -*-
"""
交互式工具测试: LearningProgress / Quiz / HintSystem
"""

import json
from pathlib import Path

import pytest

from deep_learning.utils import HintSystem, LearningProgress, Quiz


@pytest.mark.unit
def test_learning_progress_add_and_mark(tmp_path: Path):
    store = tmp_path / ".progress.json"
    lp = LearningProgress(storage_path=store)
    lp.add_task("t1", total=3)
    lp.mark_step("t1")
    lp.mark_step("t1", step=2)

    summary = lp.summary()
    assert summary["task_count"] == 1
    assert summary["completed_steps"] == 3
    assert summary["percent"] == 100.0

    # 持久化校验
    lp2 = LearningProgress(storage_path=store)
    assert lp2.summary()["percent"] == 100.0


@pytest.mark.unit
def test_learning_progress_mark_complete_creates_task(tmp_path: Path):
    store = tmp_path / ".progress.json"
    lp = LearningProgress(storage_path=store)
    lp.mark_complete("auto_task")
    summary = lp.summary()
    assert summary["task_count"] == 1
    assert summary["percent"] == 100.0


@pytest.mark.unit
def test_quiz_grading_from_sequence():
    questions = [
        {"id": "q1", "question": "1+1", "answer": "2", "points": 2},
        {"id": "q2", "question": "2+2", "answer": "4", "points": 3},
    ]
    quiz = Quiz(questions)
    result = quiz.grade(["2", "0"])  # 一个正确一个错误
    assert result["score"] == 2
    assert result["total"] == 5
    assert result["percent"] == 40.0
    details = result["details"]
    assert details[0]["correct"] is True
    assert details[1]["correct"] is False


@pytest.mark.unit
def test_quiz_grading_from_file(tmp_path: Path):
    data = [
        {"id": "q1", "question": "sky color", "answer": "blue"},
    ]
    qfile = tmp_path / "quiz.json"
    qfile.write_text(json.dumps(data), encoding="utf-8")
    quiz = Quiz(qfile)
    result = quiz.grade({"q1": "blue"})
    assert result["score"] == 1
    assert result["percent"] == 100.0


@pytest.mark.unit
def test_hint_system_levels():
    hs = HintSystem(["hint1", "hint2", "hint3"])
    assert hs.get_hint(1) == "hint1"
    assert hs.get_hint(3) == "hint3"
    # 超过最大级别时返回最后一个提示
    assert hs.get_hint(5) == "hint3"

    nxt = hs.next_hint()
    assert nxt["level"] == 1
    assert nxt["hint"] == "hint1"
    nxt2 = hs.next_hint(nxt["level"])
    assert nxt2["level"] == 2
    assert nxt2["hint"] == "hint2"
    assert nxt2["is_final"] is False
    nxt3 = hs.next_hint(nxt2["level"])
    assert nxt3["is_final"] is True
