# -*- coding: utf-8 -*-
"""
性能优化相关测试
"""

from deep_learning.utils import EarlyStopping
from deep_learning.utils import GradientAccumulator


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=2, delta=0.0, mode="min")
    scores = [0.5, 0.45, 0.46, 0.47]  # 第三次未提升后应停止
    stopped = False
    for idx, s in enumerate(scores):
        stopped = es.step(s, epoch=idx)
        if stopped:
            break
    assert stopped is True
    assert es.stopped_epoch == 3


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=3, delta=0.01, mode="max")
    assert es.step(0.5) is False  # 初次最佳
    # 小于 delta 的提升视为未提升
    assert es.step(0.505) is False
    assert es.step(0.50) is False
    assert es.step(0.49) is True  # 连续未提升触发停止


def test_gradient_accumulator_mean_and_reset():
    acc = GradientAccumulator()
    acc.add([1.0, 2.0])
    acc.add([3.0, 6.0])
    mean = acc.mean()
    assert mean == [2.0, 4.0]
    assert acc.mean() is None  # 重置后无数据
