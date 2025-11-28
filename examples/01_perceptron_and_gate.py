# -*- coding: utf-8 -*-
"""
示例 01: 感知机实现 AND 门
"""

from deep_learning.fundamentals import Perceptron
from deep_learning.utils import plot_loss_curve


def main():
    perceptron = Perceptron(input_size=2, learning_rate=0.1)

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]
    loss_history = []

    for _ in range(20):
        perceptron.train(X, y)
        # 简单统计：误分类数量作为粗略“损失”
        mis = sum(abs(perceptron.predict(xi) - yi) for xi, yi in zip(X, y))
        loss_history.append(mis)

    for xi, yi in zip(X, y):
        pred = perceptron.predict(xi)
        print(f"Input: {xi}, Predicted: {pred}, Actual: {yi}")

    # 可视化误分类数量（如果安装了 matplotlib 将显示图像，否则打印提示）
    plot_loss_curve(loss_history, title="Perceptron AND misclassification count")


if __name__ == "__main__":
    main()
