# -*- coding: utf-8 -*-
"""
Adam 优化步骤（轻量版）

简化实现，便于教学演示。
"""

from typing import List, Tuple
import math


def adam_update(params: List[float],
                grads: List[float],
                m: List[float],
                v: List[float],
                lr: float = 0.001,
                beta1: float = 0.9,
                beta2: float = 0.999,
                eps: float = 1e-8,
                t: int = 1) -> Tuple[List[float], List[float], List[float]]:
    """
    执行一次 Adam 更新。

    Args:
        params: 参数列表
        grads: 梯度列表
        m: 一阶动量
        v: 二阶动量
        lr: 学习率
        beta1: 一阶动量衰减
        beta2: 二阶动量衰减
        eps: 数值稳定项
        t: 当前步数（从1开始）

    Returns:
        (新参数, 新一阶动量, 新二阶动量)
    """
    if not (len(params) == len(grads) == len(m) == len(v)):
        raise ValueError("params、grads、m、v 长度必须一致")

    new_params, new_m, new_v = [], [], []

    for p, g, m_t, v_t in zip(params, grads, m, v):
        m_next = beta1 * m_t + (1 - beta1) * g
        v_next = beta2 * v_t + (1 - beta2) * (g ** 2)

        # 偏置校正
        m_hat = m_next / (1 - beta1 ** t)
        v_hat = v_next / (1 - beta2 ** t)

        p_next = p - lr * m_hat / (math.sqrt(v_hat) + eps)

        new_params.append(p_next)
        new_m.append(m_next)
        new_v.append(v_next)

    return new_params, new_m, new_v


__all__ = ['adam_update']
