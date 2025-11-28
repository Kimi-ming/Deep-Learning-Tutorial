# -*- coding: utf-8 -*-
"""
visualization 工具测试
"""

import types
from deep_learning.utils import visualization


def test_plot_loss_curve_no_matplotlib(monkeypatch):
    """当 matplotlib 不可用时应优雅跳过"""
    monkeypatch.setattr(visualization, "_ensure_matplotlib", lambda: None)
    visualization.plot_loss_curve([0.5, 0.3, 0.1])


def test_plot_accuracy_curve_with_dummy(monkeypatch):
    """使用假对象模拟 matplotlib，验证调用链"""

    class DummyPlot:
        def __init__(self):
            self.figure_called = False
            self.plots = []
            self.xlabel_called = False
            self.ylabel_called = False
            self.title_called = False
            self.legend_called = False
            self.show_called = False

        def figure(self):
            self.figure_called = True

        def plot(self, data, label=None):
            self.plots.append((list(data), label))

        def xlabel(self, *_):
            self.xlabel_called = True

        def ylabel(self, *_):
            self.ylabel_called = True

        def title(self, *_):
            self.title_called = True

        def legend(self, *_):
            self.legend_called = True

        def show(self, *_):
            self.show_called = True

    dummy = DummyPlot()
    monkeypatch.setattr(visualization, "_ensure_matplotlib", lambda: dummy)

    visualization.plot_accuracy_curve([0.2, 0.6, 0.9], title="acc")

    assert dummy.figure_called
    assert dummy.plots and dummy.plots[0][0] == [0.2, 0.6, 0.9]
    assert dummy.xlabel_called and dummy.ylabel_called
    assert dummy.title_called and dummy.legend_called and dummy.show_called

