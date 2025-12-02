# -*- coding: utf-8 -*-
"""
分布式训练示例（纯 Python 模拟）

说明:
- 使用 multiprocessing 启动 worker，模拟梯度求和再更新。
- 仅演示流程，不涉及真实大规模通信。
"""

import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import List, Tuple

from deep_learning.fundamentals import Perceptron


@dataclass
class Gradients:
    weights: List[float]
    bias: float


def worker(data: Tuple[List[List[float]], List[int]]) -> Gradients:
    X, y = data
    model = Perceptron(input_size=len(X[0]), learning_rate=0.1)
    model.train(X, y, epochs=5)
    return Gradients(weights=model.weights, bias=model.bias)


def average_gradients(grads: List[Gradients]) -> Gradients:
    n = len(grads)
    wlen = len(grads[0].weights)
    avg_w = [0.0] * wlen
    avg_b = 0.0
    for g in grads:
        for i, w in enumerate(g.weights):
            avg_w[i] += w / n
        avg_b += g.bias / n
    return Gradients(weights=avg_w, bias=avg_b)


def main():
    # 生成简单 AND 数据，切分成 2 份
    full_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    full_y = [0, 0, 0, 1]
    random.shuffle(full_X)
    shards = [
        (full_X[:2], full_y[:2]),
        (full_X[2:], full_y[2:]),
    ]

    with mp.Pool(processes=2) as pool:
        grads = pool.map(worker, shards)

    avg = average_gradients(grads)
    # 用平均梯度构造最终模型
    final_model = Perceptron(input_size=2, learning_rate=0.1)
    final_model.weights = avg.weights
    final_model.bias = avg.bias

    preds = [final_model.predict(x) for x in full_X]
    print("集成后的权重:", final_model.weights, "偏置:", final_model.bias)
    print("预测:", preds, "目标:", full_y)


if __name__ == "__main__":
    main()
