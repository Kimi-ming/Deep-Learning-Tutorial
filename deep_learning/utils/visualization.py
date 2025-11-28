# -*- coding: utf-8 -*-
"""
可视化工具（轻量版）

依赖:
- matplotlib (可选). 如未安装，函数会友好提示。
"""


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        print("提示: 未安装 matplotlib，跳过绘图。可通过 pip install matplotlib 安装。")
        return None


def plot_loss_curve(loss_history, title="Training Loss"):
    """绘制损失曲线"""
    plt = _ensure_matplotlib()
    if plt is None:
        return
    plt.figure()
    plt.plot(loss_history, label="loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_accuracy_curve(acc_history, title="Accuracy"):
    """绘制准确率曲线"""
    plt = _ensure_matplotlib()
    if plt is None:
        return
    plt.figure()
    plt.plot(acc_history, label="accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()


__all__ = ["plot_loss_curve", "plot_accuracy_curve"]
