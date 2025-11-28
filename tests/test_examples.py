# -*- coding: utf-8 -*-
"""
示例运行测试（确保不崩溃）
"""

import importlib
import types
from deep_learning.utils import visualization


class DummyPlot:
    def __init__(self):
        self.figure_called = False
    def figure(self): self.figure_called = True
    def plot(self, *a, **k): pass
    def xlabel(self, *a, **k): pass
    def ylabel(self, *a, **k): pass
    def title(self, *a, **k): pass
    def legend(self, *a, **k): pass
    def show(self, *a, **k): pass


def _monkey_visual(monkeypatch):
    dummy = DummyPlot()
    monkeypatch.setattr(visualization, "_ensure_matplotlib", lambda: dummy)
    return dummy


def test_examples_run(monkeypatch):
    # mock 可视化，避免依赖 matplotlib
    _monkey_visual(monkeypatch)

    for mod_name in [
        "examples.01_perceptron_and_gate",
        "examples.02_mlp_xor_gate",
        "examples.03_cnn_edge_detection",
        "examples.04_rnn_sequence_memory",
        "examples.05_transformer_attention",
    ]:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main")
        mod.main()

