# -*- coding: utf-8 -*-
"""
混合精度训练示例（纯 Python 模拟）

说明:
- 使用缩放因子模拟 FP16（权重/梯度仍用 float）。
- 展示梯度缩放和反缩放流程。
"""

import math
from typing import List

from deep_learning.fundamentals import Perceptron


def scale(tensor: List[float], factor: float) -> List[float]:
    return [x * factor for x in tensor]


def descale(tensor: List[float], factor: float) -> List[float]:
    return [x / factor for x in tensor]


def main():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    model = Perceptron(input_size=2, learning_rate=0.1)
    loss_history = []
    scale_factor = 1024.0  # 模拟 FP16 梯度缩放

    for epoch in range(5):
        for inputs, target in zip(X, y):
            pred = model.predict(inputs)
            error = target - pred
            # 模拟“梯度”
            grads_w = [error * x for x in inputs]
            grad_b = error

            # 梯度缩放
            scaled_w = scale(grads_w, scale_factor)
            scaled_b = grad_b * scale_factor

            # 反缩放并更新
            model.weights = [w + (sw / scale_factor) * model.learning_rate for w, sw in zip(model.weights, scaled_w)]
            model.bias += (scaled_b / scale_factor) * model.learning_rate

            loss = math.fabs(error)
            loss_history.append(loss)

        print(f"Epoch {epoch+1}, loss={sum(loss_history)/len(loss_history):.4f}, weights={model.weights}")

    preds = [model.predict(x) for x in X]
    print("混合精度模拟完成，预测:", preds)


if __name__ == "__main__":
    main()
