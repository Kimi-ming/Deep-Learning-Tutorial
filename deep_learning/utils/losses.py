# -*- coding: utf-8 -*-
"""
损失函数工具模块

提供常用的损失函数实现，包含数值稳定性处理。

支持的损失函数:
- mse: 均方误差（回归）
- mae: 平均绝对误差（回归）
- binary_cross_entropy: 二分类交叉熵
- categorical_cross_entropy: 多分类交叉熵
- hinge: 支持向量机损失
"""

import math


def mse_loss(y_true, y_pred):
    """
    均方误差损失（Mean Squared Error）

    数学公式: MSE = (1/n) * Σ(y_true - y_pred)²

    用途:
    - 回归问题
    - 连续值预测

    特点:
    - 对异常值敏感（平方放大误差）
    - 可微分，易于优化
    - 梯度: ∂MSE/∂y_pred = 2(y_pred - y_true) / n

    Args:
        y_true: 真实值列表
        y_pred: 预测值列表

    Returns:
        均方误差值

    Examples:
        >>> mse_loss([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        0.01333...
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")

    n = len(y_true)
    squared_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]
    return sum(squared_errors) / n


def mae_loss(y_true, y_pred):
    """
    平均绝对误差损失（Mean Absolute Error）

    数学公式: MAE = (1/n) * Σ|y_true - y_pred|

    用途:
    - 回归问题
    - 对异常值更鲁棒（相比MSE）

    特点:
    - 不受异常值过度影响
    - 梯度恒定（不依赖误差大小）
    - 梯度: ∂MAE/∂y_pred = sign(y_pred - y_true) / n

    Args:
        y_true: 真实值列表
        y_pred: 预测值列表

    Returns:
        平均绝对误差值

    Examples:
        >>> mae_loss([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        0.1
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")

    n = len(y_true)
    absolute_errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
    return sum(absolute_errors) / n


def binary_cross_entropy(y_true, y_pred, epsilon=1e-7):
    """
    二分类交叉熵损失（Binary Cross-Entropy）

    数学公式: BCE = -[y*log(p) + (1-y)*log(1-p)]

    用途:
    - 二分类问题
    - 输出层使用sigmoid激活

    数值稳定性:
    - 添加epsilon避免log(0)
    - 裁剪预测值到 [epsilon, 1-epsilon]

    Args:
        y_true: 真实标签列表（0或1）
        y_pred: 预测概率列表（0到1之间）
        epsilon: 数值稳定性参数

    Returns:
        二分类交叉熵损失值

    Examples:
        >>> binary_cross_entropy([1, 0, 1], [0.9, 0.1, 0.8])
        0.174...
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")

    n = len(y_true)
    total_loss = 0.0

    for yt, yp in zip(y_true, y_pred):
        # 裁剪预测值，避免log(0)
        yp_clipped = max(epsilon, min(1 - epsilon, yp))

        # 计算交叉熵
        loss = -(yt * math.log(yp_clipped) + (1 - yt) * math.log(1 - yp_clipped))
        total_loss += loss

    return total_loss / n


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-7):
    """
    多分类交叉熵损失（Categorical Cross-Entropy）

    数学公式: CCE = -Σ y_true * log(y_pred)

    用途:
    - 多分类问题
    - 输出层使用softmax激活

    数值稳定性:
    - 添加epsilon避免log(0)
    - y_true应该是one-hot编码

    Args:
        y_true: 真实标签列表（one-hot编码或标签列表）
        y_pred: 预测概率分布列表
        epsilon: 数值稳定性参数

    Returns:
        多分类交叉熵损失值

    Examples:
        >>> # y_true是one-hot: [1, 0, 0]表示类别0
        >>> categorical_cross_entropy([1, 0, 0], [0.7, 0.2, 0.1])
        0.356...
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")

    total_loss = 0.0

    for yt, yp in zip(y_true, y_pred):
        # 裁剪预测值，避免log(0)
        yp_clipped = max(epsilon, min(1.0, yp))
        total_loss += yt * math.log(yp_clipped)

    return -total_loss


def sparse_categorical_cross_entropy(y_true_indices, y_pred_probs, epsilon=1e-7):
    """
    稀疏多分类交叉熵损失

    与categorical_cross_entropy类似，但y_true是类别索引而非one-hot

    Args:
        y_true_indices: 真实类别索引列表 [0, 2, 1, ...]
        y_pred_probs: 预测概率分布列表 [[p0, p1, p2], ...]
        epsilon: 数值稳定性参数

    Returns:
        平均交叉熵损失值

    Examples:
        >>> # 样本0属于类别1，预测概率[0.1, 0.7, 0.2]
        >>> sparse_categorical_cross_entropy([1], [[0.1, 0.7, 0.2]])
        0.356...
    """
    if len(y_true_indices) != len(y_pred_probs):
        raise ValueError("y_true和y_pred的批次大小不匹配")

    n = len(y_true_indices)
    total_loss = 0.0

    for true_idx, pred_probs in zip(y_true_indices, y_pred_probs):
        # 获取真实类别的预测概率
        prob = pred_probs[true_idx]

        # 裁剪并计算log
        prob_clipped = max(epsilon, min(1.0, prob))
        total_loss += -math.log(prob_clipped)

    return total_loss / n


def hinge_loss(y_true, y_pred):
    """
    Hinge损失（支持向量机损失）

    数学公式: L = max(0, 1 - y_true * y_pred)

    用途:
    - 支持向量机（SVM）
    - 最大间隔分类

    特点:
    - y_true ∈ {-1, +1}
    - 鼓励正确分类且间隔≥1
    - 非可微（在y*f(x)=1处）

    Args:
        y_true: 真实标签列表（-1或+1）
        y_pred: 预测得分列表（实数）

    Returns:
        Hinge损失值

    Examples:
        >>> hinge_loss([1, -1, 1], [0.8, -0.9, 1.5])
        0.1
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")

    n = len(y_true)
    losses = [max(0, 1 - yt * yp) for yt, yp in zip(y_true, y_pred)]
    return sum(losses) / n


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber损失（结合MSE和MAE的优点）

    数学公式:
    - |error| ≤ δ: L = 0.5 * error²
    - |error| > δ: L = δ * (|error| - 0.5*δ)

    用途:
    - 回归问题
    - 对异常值更鲁棒（相比MSE）
    - 接近0时更平滑（相比MAE）

    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        delta: 阈值参数

    Returns:
        Huber损失值

    Examples:
        >>> huber_loss([1.0, 2.0], [1.1, 2.5], delta=1.0)
        0.405
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}")

    n = len(y_true)
    total_loss = 0.0

    for yt, yp in zip(y_true, y_pred):
        error = abs(yt - yp)

        if error <= delta:
            # 类似MSE
            loss = 0.5 * error * error
        else:
            # 类似MAE
            loss = delta * (error - 0.5 * delta)

        total_loss += loss

    return total_loss / n


# 损失函数字典（方便动态调用）
LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'mae': mae_loss,
    'binary_cross_entropy': binary_cross_entropy,
    'categorical_cross_entropy': categorical_cross_entropy,
    'sparse_categorical_cross_entropy': sparse_categorical_cross_entropy,
    'hinge': hinge_loss,
    'huber': huber_loss,
}


def get_loss_function(name):
    """
    根据名称获取损失函数

    Args:
        name: 损失函数名称

    Returns:
        损失函数

    Raises:
        ValueError: 如果损失函数不存在
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"未知的损失函数: {name}. 可用: {list(LOSS_FUNCTIONS.keys())}")
    return LOSS_FUNCTIONS[name]
