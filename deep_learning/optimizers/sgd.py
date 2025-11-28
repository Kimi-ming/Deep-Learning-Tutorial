# -*- coding: utf-8 -*-
"""
SGD 优化步骤（轻量版）

提供基础的随机梯度下降更新，用于教学示例。
"""

from typing import List


def sgd_step(params: List[float], grads: List[float], lr: float = 0.01):
    """
    对参数执行一次 SGD 更新。

    Args:
        params: 参数列表
        grads: 梯度列表（与 params 对齐）
        lr: 学习率

    Returns:
        更新后的参数列表
    """
    if len(params) != len(grads):
        raise ValueError("params 与 grads 长度不一致")

    return [p - lr * g for p, g in zip(params, grads)]


__all__ = ['sgd_step']
