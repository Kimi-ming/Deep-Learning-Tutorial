# -*- coding: utf-8 -*-
"""
示例 02: MLP 近似 XOR 门（演示网络调用）
"""

from deep_learning.fundamentals import MLP
from deep_learning.utils import plot_loss_curve, plot_accuracy_curve


def main():
    # 简单两层 MLP
    mlp = MLP(layers=[2, 4, 1], learning_rate=0.1, activation="relu")

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]
    loss_history = []
    acc_history = []

    # 仅演示前向输出，不包含完整反向传播训练
    for xi, yi in zip(X, y):
        preds, _ = mlp.forward(xi, training=False)
        print(f"Input: {xi}, Pred: {preds[-1]}, Target: {yi}")
        # 记录一个简单的 L1 误差与准确率估计
        loss = sum(abs(a - b) for a, b in zip(preds[-1], yi))
        loss_history.append(loss)
        acc_history.append(int(round(preds[-1][0]) == yi[0]))

    plot_loss_curve(loss_history, title="MLP XOR demo loss (L1)")
    plot_accuracy_curve(acc_history, title="MLP XOR demo acc (rounded)")


if __name__ == "__main__":
    main()
