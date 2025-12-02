# -*- coding: utf-8 -*-
"""
交互式学习工具 / Interactive learning helpers

提供:
- LearningProgress: 进度跟踪与持久化 (.progress.json)
- Quiz: 题库加载与评分
- HintSystem: 分级提示管理（默认三层）

English:
- LearningProgress: track progress with JSON persistence
- Quiz: load questions and grade responses
- HintSystem: multi-level hints (3 levels by default)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


class LearningProgress:
    """
    简单的进度追踪器 / Simple progress tracker

    数据结构 / Data layout:
    {
        "tasks": {
            "task_id": {"completed": int, "total": int}
        }
    }
    """

    def __init__(self, storage_path: Union[str, Path] = ".progress.json"):
        self.storage_path = Path(storage_path)
        self.data: Dict[str, Dict[str, Dict[str, int]]] = {"tasks": {}}
        self.load()

    def load(self) -> None:
        if self.storage_path.exists():
            self.data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        else:
            self.save()

    def save(self) -> None:
        self.storage_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_task(self, task_id: str, total: int = 1) -> None:
        if total <= 0:
            raise ValueError("total 必须为正整数")
        self.data["tasks"].setdefault(task_id, {"completed": 0, "total": total})
        self.save()

    def mark_step(self, task_id: str, step: int = 1) -> None:
        if step <= 0:
            raise ValueError("step 必须为正整数")
        task = self.data["tasks"].get(task_id)
        if task is None:
            task = {"completed": 0, "total": step}
            self.data["tasks"][task_id] = task
        task["completed"] = min(task["completed"] + step, task["total"])
        self.save()

    def mark_complete(self, task_id: str) -> None:
        task = self.data["tasks"].get(task_id)
        if task is None:
            task = {"completed": 1, "total": 1}
            self.data["tasks"][task_id] = task
        else:
            task["completed"] = task["total"]
        self.save()

    def summary(self) -> Dict[str, Union[int, float]]:
        tasks = self.data.get("tasks", {})
        total_items = len(tasks)
        total_steps = sum(t["total"] for t in tasks.values()) or 1
        completed_steps = sum(t["completed"] for t in tasks.values())
        percent = round((completed_steps / total_steps) * 100, 2)
        return {
            "task_count": total_items,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "percent": percent,
        }

    def percent(self) -> float:
        return self.summary()["percent"]  # type: ignore[return-value]


class Quiz:
    """
    简单测验系统 / Lightweight quiz system

    题目格式 / Question format:
    {
        "id": "q1",
        "question": "1+1=?",
        "options": ["1", "2", "3"],
        "answer": "2",
        "points": 1
    }
    """

    def __init__(self, questions: Union[str, Path, Sequence[Dict[str, Union[str, int, List[str]]]]]):
        if isinstance(questions, (str, Path)):
            questions_path = Path(questions)
            loaded = json.loads(questions_path.read_text(encoding="utf-8"))
        else:
            loaded = list(questions)
        self.questions: List[Dict[str, Union[str, int, List[str]]]] = []
        for idx, q in enumerate(loaded):
            cleaned = dict(q)
            cleaned.setdefault("id", f"q{idx+1}")
            cleaned.setdefault("points", 1)
            self.questions.append(cleaned)

    def grade(self, responses: Union[Sequence[str], Dict[str, str]]) -> Dict[str, Union[int, float, List[Dict[str, Union[str, bool, int]]]]]:
        if isinstance(responses, dict):
            response_lookup = responses
        else:
            response_lookup = {}
            for q, resp in zip(self.questions, responses):
                response_lookup[str(q["id"])] = resp

        details = []
        total_points = 0
        earned_points = 0

        for q in self.questions:
            qid = str(q["id"])
            correct = q.get("answer")
            points = int(q.get("points", 1))
            total_points += points
            user_answer = response_lookup.get(qid)
            is_correct = user_answer == correct
            if is_correct:
                earned_points += points
            details.append({
                "id": qid,
                "correct": is_correct,
                "points": points,
                "earned": points if is_correct else 0,
                "user_answer": user_answer,
                "answer": correct,
            })

        percent = round((earned_points / total_points) * 100, 2) if total_points else 0.0

        return {
            "score": earned_points,
            "total": total_points,
            "percent": percent,
            "details": details,
        }


class HintSystem:
    """
    分级提示系统（默认三级）/ Multi-level hint system (3 levels by default)
    """

    def __init__(self, hints: Sequence[str]):
        if not hints:
            raise ValueError("必须提供至少一个提示")
        self.hints: List[str] = list(hints)

    def get_hint(self, level: int = 1) -> str:
        if level <= 0:
            raise ValueError("level 必须为正整数")
        idx = min(level, len(self.hints)) - 1
        return self.hints[idx]

    def next_hint(self, current_level: Optional[int] = None) -> Dict[str, Union[int, str, bool]]:
        """
        获取下一个层级的提示，并告知是否已到最后一级。
        """
        level = 1 if current_level is None else current_level + 1
        hint = self.get_hint(level)
        return {
            "level": level,
            "hint": hint,
            "is_final": level >= len(self.hints),
        }


__all__ = ["LearningProgress", "Quiz", "HintSystem"]
