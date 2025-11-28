# -*- coding: utf-8 -*-
"""
基础模块子包

提供感知机与多层感知机等基础网络的包装，便于后续迁移。
"""

from .perceptron import Perceptron
from .deep_network import DeepNetwork
from .mlp import MLP

# 兼容旧版脚本中的教学函数，延迟导入以避免循环
def deep_learning_introduction():
    import importlib
    legacy = importlib.import_module("deep_learning_fundamentals")
    return legacy.deep_learning_introduction()


def deep_learning_architectures_overview():
    import importlib
    legacy = importlib.import_module("deep_learning_fundamentals")
    return legacy.deep_learning_architectures_overview()


def main():
    import importlib
    legacy = importlib.import_module("deep_learning_fundamentals")
    return legacy.main()

__all__ = [
    'Perceptron', 'MLP', 'DeepNetwork',
    'deep_learning_introduction', 'deep_learning_architectures_overview', 'main'
]
