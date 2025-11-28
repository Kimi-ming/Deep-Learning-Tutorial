# -*- coding: utf-8 -*-
"""
练习模块包装

提供对旧版 `deep_learning_exercises.py` 的兼容访问，并为后续重构预留入口。
"""

import importlib

# 兼容：导出旧练习函数的简单调用包装
_legacy = importlib.import_module("deep_learning_exercises")


def list_exercises():
    """列出可用练习函数名（基于 legacy 模块前缀 `exercise_`）"""
    return [name for name in dir(_legacy) if name.startswith("exercise_")]


def run_exercise(name: str):
    """
    运行指定练习（如 'exercise_1_perceptron'）
    """
    fn = getattr(_legacy, name, None)
    if fn is None:
        raise ValueError(f"未找到练习: {name}")
    return fn()


__all__ = ["list_exercises", "run_exercise"]
