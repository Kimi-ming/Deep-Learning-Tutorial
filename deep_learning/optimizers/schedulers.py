# -*- coding: utf-8 -*-
"""
学习率调度器（轻量版）
"""

import math


def step_decay(initial_lr: float, epoch: int, drop_every: int = 10, drop_rate: float = 0.5):
    """阶梯衰减学习率"""
    factor = drop_rate ** (epoch // drop_every)
    return initial_lr * factor


def cosine_decay(initial_lr: float, step: int, total_steps: int):
    """余弦退火学习率"""
    if total_steps <= 0:
        raise ValueError("total_steps 必须大于 0")
    cosine_term = 0.5 * (1 + math.cos(math.pi * step / total_steps))
    return initial_lr * cosine_term


__all__ = ['step_decay', 'cosine_decay']
