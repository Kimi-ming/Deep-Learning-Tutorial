# -*- coding: utf-8 -*-
"""
模型保存/加载测试
"""

from pathlib import Path

from deep_learning.fundamentals import Perceptron
from deep_learning.utils import load_model, save_model


def _make_trained_perceptron():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]
    model = Perceptron(input_size=2, learning_rate=0.2)
    model.train(X, y, epochs=5)
    return model


def test_pickle_roundtrip(tmp_path: Path):
    model = _make_trained_perceptron()
    path = tmp_path / "model.pkl"
    save_model(model, path, fmt="pickle")
    loaded = load_model(path, fmt="pickle")

    assert isinstance(loaded, Perceptron)
    assert loaded.weights == model.weights
    assert loaded.bias == model.bias


def test_json_roundtrip(tmp_path: Path):
    model = _make_trained_perceptron()
    path = tmp_path / "model.json"
    save_model(model, path, fmt="json")
    loaded = load_model(path, fmt="json")

    assert isinstance(loaded, Perceptron)
    assert loaded.weights == model.weights
    assert loaded.bias == model.bias
