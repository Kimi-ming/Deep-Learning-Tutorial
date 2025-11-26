# -*- coding: utf-8 -*-
"""
激活函数工具模块

提供常用的激活函数实现，包含数值稳定性处理。

支持的激活函数:
- sigmoid: S型函数，输出范围 (0, 1)
- tanh: 双曲正切函数，输出范围 (-1, 1)
- relu: 修正线性单元，输出范围 [0, +∞)
- leaky_relu: 带泄漏的ReLU，避免死亡神经元
- softmax: 将logits转换为概率分布
"""

import math


def sigmoid(x):
    """
    Sigmoid 激活函数（数值稳定版本）

    数学公式: σ(x) = 1 / (1 + e^(-x))

    特点:
    - 输出范围: (0, 1)
    - 梯度: σ'(x) = σ(x) * (1 - σ(x))
    - 问题: 梯度消失（两端梯度接近0）

    数值稳定性处理:
    - 对于 x > 500: 返回 1.0 (避免 exp 溢出)
    - 对于 x < -500: 返回 0.0 (避免 exp 溢出)
    - 对于 x < 0: 使用 e^x / (1 + e^x) 形式

    Args:
        x: 输入值（标量）

    Returns:
        激活后的值，范围在 (0, 1)

    Examples:
        >>> sigmoid(0)
        0.5
        >>> sigmoid(100)  # 大正数
        1.0
        >>> sigmoid(-100)  # 大负数
        0.0
    """
    # 数值稳定性：防止 exp 溢出
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0

    # 对于负数，使用等价形式避免数值问题
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def tanh(x):
    """
    Tanh 激活函数（双曲正切）

    数学公式: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    特点:
    - 输出范围: (-1, 1)
    - 梯度: tanh'(x) = 1 - tanh²(x)
    - 零中心化（相比sigmoid的优势）
    - 仍存在梯度消失问题

    数值稳定性处理:
    - 对于 |x| > 20: 返回 ±1.0

    Args:
        x: 输入值（标量）

    Returns:
        激活后的值，范围在 (-1, 1)

    Examples:
        >>> tanh(0)
        0.0
        >>> abs(tanh(10) - 1.0) < 0.01
        True
        >>> abs(tanh(-10) - (-1.0)) < 0.01
        True
    """
    # 数值稳定性：对于极大/极小值直接返回
    if x > 20:
        return 1.0
    elif x < -20:
        return -1.0

    return math.tanh(x)


def relu(x):
    """
    ReLU 激活函数（修正线性单元）

    数学公式: ReLU(x) = max(0, x)

    特点:
    - 输出范围: [0, +∞)
    - 梯度: 1 (x > 0), 0 (x ≤ 0)
    - 优点: 计算简单，缓解梯度消失
    - 问题: 死亡ReLU（负值神经元永不激活）

    Args:
        x: 输入值（标量）

    Returns:
        激活后的值，范围在 [0, +∞)

    Examples:
        >>> relu(5.0)
        5.0
        >>> relu(-3.0)
        0.0
        >>> relu(0.0)
        0.0
    """
    return max(0.0, x)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU 激活函数

    数学公式: LeakyReLU(x) = max(αx, x) where α ∈ (0, 1)

    特点:
    - 输出范围: (-∞, +∞)
    - 梯度: 1 (x > 0), α (x ≤ 0)
    - 解决死亡ReLU问题（负值也有小梯度）

    Args:
        x: 输入值（标量）
        alpha: 负半轴的斜率，默认 0.01

    Returns:
        激活后的值

    Examples:
        >>> leaky_relu(5.0)
        5.0
        >>> leaky_relu(-10.0)
        -0.1
        >>> leaky_relu(-10.0, alpha=0.2)
        -2.0
    """
    return max(alpha * x, x)


def softmax(logits):
    """
    Softmax 激活函数（数值稳定版本）

    数学公式: softmax(x_i) = e^(x_i) / Σ(e^(x_j))

    特点:
    - 输出: 概率分布，和为1
    - 用途: 多分类问题的输出层
    - 可微分，便于反向传播

    数值稳定性处理:
    - 减去最大值: softmax(x - max(x)) = softmax(x)
    - 避免 exp 溢出

    Args:
        logits: 输入列表/向量（未归一化的对数概率）

    Returns:
        概率分布列表，所有元素和为1

    Examples:
        >>> probs = softmax([1.0, 2.0, 3.0])
        >>> abs(sum(probs) - 1.0) < 1e-6
        True
        >>> all(0 <= p <= 1 for p in probs)
        True
    """
    # 数值稳定性：减去最大值
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)

    # 避免除以0
    if sum_exp == 0:
        # 均匀分布
        return [1.0 / len(logits) for _ in logits]

    return [exp_val / sum_exp for exp_val in exp_logits]


def sigmoid_derivative(x):
    """
    Sigmoid 函数的导数

    数学公式: σ'(x) = σ(x) * (1 - σ(x))

    Args:
        x: 输入值（通常是sigmoid的输出）

    Returns:
        导数值
    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x):
    """
    Tanh 函数的导数

    数学公式: tanh'(x) = 1 - tanh²(x)

    Args:
        x: 输入值（通常是tanh的输出）

    Returns:
        导数值
    """
    t = tanh(x)
    return 1 - t * t


def relu_derivative(x):
    """
    ReLU 函数的导数

    数学公式: ReLU'(x) = 1 if x > 0 else 0

    Args:
        x: 输入值

    Returns:
        导数值（0 或 1）
    """
    return 1.0 if x > 0 else 0.0


# 激活函数字典（方便动态调用）
ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'softmax': softmax,
}

# 激活函数导数字典
ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
}


def get_activation(name):
    """
    根据名称获取激活函数

    Args:
        name: 激活函数名称

    Returns:
        激活函数

    Raises:
        ValueError: 如果激活函数不存在
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"未知的激活函数: {name}. 可用: {list(ACTIVATION_FUNCTIONS.keys())}")
    return ACTIVATION_FUNCTIONS[name]


def get_activation_derivative(name):
    """
    根据名称获取激活函数的导数

    Args:
        name: 激活函数名称

    Returns:
        激活函数的导数

    Raises:
        ValueError: 如果导数不存在
    """
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"未知的激活函数导数: {name}. 可用: {list(ACTIVATION_DERIVATIVES.keys())}")
    return ACTIVATION_DERIVATIVES[name]
