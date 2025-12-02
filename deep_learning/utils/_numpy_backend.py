# -*- coding: utf-8 -*-
"""
NumPy 加速后端

提供若干运算的 NumPy 实现，并暴露 is_available() 便于运行时检测。

English:
- NumPy-backed implementations for selected ops.
- Use `is_available()` to check runtime availability.
"""

from __future__ import annotations

try:
    import numpy as _np
except Exception:  # pragma: no cover - NumPy 应该存在，但保留安全路径
    _np = None  # type: ignore


def is_available() -> bool:
    """NumPy 是否可用"""
    return _np is not None


def matmul(A, B):
    """
    使用 NumPy 的矩阵乘法。
    返回 ndarray；调用侧可视需要 .tolist() 转为 Python 列表。
    """
    if _np is None:
        raise ImportError("NumPy 不可用，无法使用加速后端")
    return _np.matmul(_np.array(A), _np.array(B))


def conv2d_single_channel(input_map, kernel, bias=0.0, stride=1):
    """
    单通道 2D 卷积 (valid) 的 NumPy 实现。

    Args:
        input_map (list[list[float]]): 输入特征图
        kernel (list[list[float]]): 卷积核
        bias (float): 偏置
        stride (int): 步长

    Returns:
        numpy.ndarray: 卷积结果
    """
    if _np is None:
        raise ImportError("NumPy 不可用，无法使用加速后端")
    if stride <= 0:
        raise ValueError("stride 必须为正整数")

    x = _np.array(input_map, dtype=float)
    k = _np.array(kernel, dtype=float)

    h, w = x.shape
    kh, kw = k.shape
    if kh > h or kw > w:
        raise ValueError("卷积核尺寸不可超过输入尺寸")

    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = _np.empty((out_h, out_w), dtype=float)

    for i in range(out_h):
        for j in range(out_w):
            region = x[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = (region * k).sum() + bias

    return out


__all__ = ["is_available", "matmul", "conv2d_single_channel"]
