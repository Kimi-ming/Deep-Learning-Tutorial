# -*- coding: utf-8 -*-
"""
感知机实现（轻量包装版）

说明:
- 该实现为简单的二分类感知机，方便在迁移阶段先提供可用 API。
- 后续可替换为更完整的教学版实现，或调用 deep_learning_fundamentals 中的相关逻辑。
"""

from typing import List


class Perceptron:
    """二分类感知机"""

    def __init__(self, input_size: int, learning_rate: float = 0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        # 初始化权重和偏置
        self.weights = [0.0] * input_size
        self.bias = 0.0

    def _activate(self, x: float) -> int:
        """阶跃函数"""
        return 1 if x >= 0 else 0

    def predict(self, inputs: List[float]) -> int:
        """前向预测"""
        if len(inputs) != self.input_size:
            raise ValueError(f"输入长度应为 {self.input_size}, 实际 {len(inputs)}")
        summation = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self._activate(summation)

    def train(self, training_data, labels, epochs: int = 10):
        """感知机训练"""
        for _ in range(epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # 权重和偏置更新
                self.weights = [
                    w + self.learning_rate * error * x
                    for w, x in zip(self.weights, inputs)
                ]
                self.bias += self.learning_rate * error


__all__ = ['Perceptron']
