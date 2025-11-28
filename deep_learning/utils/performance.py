# -*- coding: utf-8 -*-
"""
性能与训练控制工具

当前提供:
- EarlyStopping: 基于验证指标的早停控制
"""

from typing import Optional


class EarlyStopping:
    """
    简单的早停器

    Args:
        patience: 连续多少次没有提升后触发停止
        delta: 提升的最小阈值
        mode: 'min' 或 'max'，决定提升方向
    """

    def __init__(self, patience: int = 3, delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError("mode 必须是 'min' 或 'max'")
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score: Optional[float] = None
        self.counter = 0
        self.stopped_epoch: Optional[int] = None
        self.should_stop = False

    def _is_improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.delta
        return score > self.best_score + self.delta

    def step(self, score: float, epoch: Optional[int] = None) -> bool:
        """
        更新早停状态。

        Returns:
            bool: 是否应停止训练
        """
        if self._is_improved(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
        return self.should_stop

    # 兼容 __call__
    __call__ = step


class GradientAccumulator:
    """
    梯度累积器：用于模拟大 batch 训练，将多次小批梯度求和/平均。
    """

    def __init__(self):
        self.storage = None
        self.count = 0

    def add(self, grads):
        """
        累加梯度（结构需一致，如 list/tuple/nested list）
        """
        if self.storage is None:
            self.storage = self._clone(grads)
        else:
            self.storage = self._accumulate(self.storage, grads)
        self.count += 1

    def mean(self):
        """返回平均梯度，并重置计数。"""
        if self.storage is None or self.count == 0:
            return None
        scaled = self._scale(self.storage, 1.0 / self.count)
        self.storage = None
        self.count = 0
        return scaled

    # 内部工具
    def _clone(self, obj):
        if isinstance(obj, (list, tuple)):
            return [self._clone(x) for x in obj]
        return float(obj)

    def _accumulate(self, target, source):
        if isinstance(target, list):
            return [self._accumulate(t, s) for t, s in zip(target, source)]
        return target + float(source)

    def _scale(self, obj, factor):
        if isinstance(obj, list):
            return [self._scale(x, factor) for x in obj]
        return obj * factor


__all__ = ["EarlyStopping"]
