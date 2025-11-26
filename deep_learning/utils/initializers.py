# -*- coding: utf-8 -*-
"""
权重初始化工具模块

提供常用的神经网络权重初始化方法。

支持的初始化方法:
- zeros: 全零初始化
- ones: 全1初始化
- uniform: 均匀分布初始化
- normal: 正态分布初始化
- xavier/glorot: Xavier初始化（适用于sigmoid/tanh）
- he: He初始化（适用于ReLU）
- lecun: LeCun初始化（适用于SELU）
"""

import random
import math


def zeros(shape):
    """
    全零初始化

    用途:
    - 偏置初始化（常用）
    - 不适合权重（会导致对称性问题）

    Args:
        shape: 形状元组，如 (rows, cols) 或 (size,)

    Returns:
        初始化后的权重矩阵/向量

    Examples:
        >>> w = zeros((2, 3))
        >>> len(w), len(w[0])
        (2, 3)
        >>> all(all(x == 0 for x in row) for row in w)
        True
    """
    if isinstance(shape, int):
        return [0.0 for _ in range(shape)]
    elif len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError(f"不支持的形状: {shape}")


def ones(shape):
    """
    全1初始化

    用途:
    - 某些特殊场景（如批归一化的gamma参数）
    - 一般不用于权重初始化

    Args:
        shape: 形状元组

    Returns:
        初始化后的权重矩阵/向量
    """
    if isinstance(shape, int):
        return [1.0 for _ in range(shape)]
    elif len(shape) == 1:
        return [1.0 for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError(f"不支持的形状: {shape}")


def uniform(shape, low=-0.1, high=0.1):
    """
    均匀分布初始化

    从均匀分布 U(low, high) 中采样

    Args:
        shape: 形状元组
        low: 下界
        high: 上界

    Returns:
        初始化后的权重矩阵/向量

    Examples:
        >>> w = uniform((2, 3), -1, 1)
        >>> all(all(-1 <= x <= 1 for x in row) for row in w)
        True
    """
    if isinstance(shape, int):
        return [random.uniform(low, high) for _ in range(shape)]
    elif len(shape) == 1:
        return [random.uniform(low, high) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.uniform(low, high) for _ in range(shape[1])]
                for _ in range(shape[0])]
    else:
        raise ValueError(f"不支持的形状: {shape}")


def normal(shape, mean=0.0, std=0.01):
    """
    正态分布初始化

    从正态分布 N(mean, std²) 中采样

    Args:
        shape: 形状元组
        mean: 均值
        std: 标准差

    Returns:
        初始化后的权重矩阵/向量

    Examples:
        >>> w = normal((100,), mean=0, std=1)
        >>> abs(sum(w)/len(w)) < 0.2  # 均值接近0
        True
    """
    if isinstance(shape, int):
        return [random.gauss(mean, std) for _ in range(shape)]
    elif len(shape) == 1:
        return [random.gauss(mean, std) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.gauss(mean, std) for _ in range(shape[1])]
                for _ in range(shape[0])]
    else:
        raise ValueError(f"不支持的形状: {shape}")


def xavier_uniform(shape, gain=1.0):
    """
    Xavier/Glorot 均匀分布初始化

    数学公式: W ~ U(-limit, limit)
    其中 limit = gain * sqrt(6 / (fan_in + fan_out))

    用途:
    - sigmoid 激活函数
    - tanh 激活函数

    原理:
    - 保持前向和反向传播时方差稳定
    - fan_in: 输入单元数
    - fan_out: 输出单元数

    Args:
        shape: 形状元组 (fan_in, fan_out)
        gain: 增益因子（默认1.0）

    Returns:
        初始化后的权重矩阵

    Examples:
        >>> w = xavier_uniform((100, 50))
        >>> len(w), len(w[0])
        (100, 50)

    References:
        Glorot & Bengio, 2010: "Understanding the difficulty of training deep feedforward neural networks"
    """
    if isinstance(shape, int):
        # 向量形式，假设fan_in=fan_out=shape
        limit = gain * math.sqrt(6.0 / (2 * shape))
        return [random.uniform(-limit, limit) for _ in range(shape)]
    elif len(shape) == 2:
        fan_in, fan_out = shape
        limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return [[random.uniform(-limit, limit) for _ in range(fan_out)]
                for _ in range(fan_in)]
    else:
        raise ValueError(f"Xavier初始化仅支持2D形状，得到: {shape}")


def xavier_normal(shape, gain=1.0):
    """
    Xavier/Glorot 正态分布初始化

    数学公式: W ~ N(0, std²)
    其中 std = gain * sqrt(2 / (fan_in + fan_out))

    Args:
        shape: 形状元组 (fan_in, fan_out)
        gain: 增益因子（默认1.0）

    Returns:
        初始化后的权重矩阵
    """
    if isinstance(shape, int):
        std = gain * math.sqrt(2.0 / (2 * shape))
        return [random.gauss(0, std) for _ in range(shape)]
    elif len(shape) == 2:
        fan_in, fan_out = shape
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return [[random.gauss(0, std) for _ in range(fan_out)]
                for _ in range(fan_in)]
    else:
        raise ValueError(f"Xavier初始化仅支持2D形状，得到: {shape}")


def he_uniform(shape):
    """
    He均匀分布初始化（Kaiming初始化）

    数学公式: W ~ U(-limit, limit)
    其中 limit = sqrt(6 / fan_in)

    用途:
    - ReLU 激活函数
    - Leaky ReLU 激活函数

    原理:
    - 针对ReLU的非对称性设计
    - 仅考虑fan_in（因为ReLU会kill一半神经元）

    Args:
        shape: 形状元组 (fan_in, fan_out)

    Returns:
        初始化后的权重矩阵

    Examples:
        >>> w = he_uniform((100, 50))
        >>> len(w), len(w[0])
        (100, 50)

    References:
        He et al., 2015: "Delving Deep into Rectifiers"
    """
    if isinstance(shape, int):
        limit = math.sqrt(6.0 / shape)
        return [random.uniform(-limit, limit) for _ in range(shape)]
    elif len(shape) == 2:
        fan_in, fan_out = shape
        limit = math.sqrt(6.0 / fan_in)
        return [[random.uniform(-limit, limit) for _ in range(fan_out)]
                for _ in range(fan_in)]
    else:
        raise ValueError(f"He初始化仅支持2D形状，得到: {shape}")


def he_normal(shape):
    """
    He正态分布初始化

    数学公式: W ~ N(0, std²)
    其中 std = sqrt(2 / fan_in)

    用途:
    - ReLU 激活函数（推荐）
    - 相比He uniform更常用

    Args:
        shape: 形状元组 (fan_in, fan_out)

    Returns:
        初始化后的权重矩阵
    """
    if isinstance(shape, int):
        std = math.sqrt(2.0 / shape)
        return [random.gauss(0, std) for _ in range(shape)]
    elif len(shape) == 2:
        fan_in, fan_out = shape
        std = math.sqrt(2.0 / fan_in)
        return [[random.gauss(0, std) for _ in range(fan_out)]
                for _ in range(fan_in)]
    else:
        raise ValueError(f"He初始化仅支持2D形状，得到: {shape}")


def lecun_uniform(shape):
    """
    LeCun均匀分布初始化

    数学公式: W ~ U(-limit, limit)
    其中 limit = sqrt(3 / fan_in)

    用途:
    - SELU 激活函数
    - 经典初始化方法

    Args:
        shape: 形状元组 (fan_in, fan_out)

    Returns:
        初始化后的权重矩阵
    """
    if isinstance(shape, int):
        limit = math.sqrt(3.0 / shape)
        return [random.uniform(-limit, limit) for _ in range(shape)]
    elif len(shape) == 2:
        fan_in, fan_out = shape
        limit = math.sqrt(3.0 / fan_in)
        return [[random.uniform(-limit, limit) for _ in range(fan_out)]
                for _ in range(fan_in)]
    else:
        raise ValueError(f"LeCun初始化仅支持2D形状，得到: {shape}")


def lecun_normal(shape):
    """
    LeCun正态分布初始化

    数学公式: W ~ N(0, std²)
    其中 std = sqrt(1 / fan_in)

    Args:
        shape: 形状元组 (fan_in, fan_out)

    Returns:
        初始化后的权重矩阵
    """
    if isinstance(shape, int):
        std = math.sqrt(1.0 / shape)
        return [random.gauss(0, std) for _ in range(shape)]
    elif len(shape) == 2:
        fan_in, fan_out = shape
        std = math.sqrt(1.0 / fan_in)
        return [[random.gauss(0, std) for _ in range(fan_out)]
                for _ in range(fan_in)]
    else:
        raise ValueError(f"LeCun初始化仅支持2D形状，得到: {shape}")


# 初始化器字典
INITIALIZERS = {
    'zeros': zeros,
    'ones': ones,
    'uniform': uniform,
    'normal': normal,
    'xavier_uniform': xavier_uniform,
    'xavier_normal': xavier_normal,
    'glorot_uniform': xavier_uniform,  # 别名
    'glorot_normal': xavier_normal,    # 别名
    'he_uniform': he_uniform,
    'he_normal': he_normal,
    'lecun_uniform': lecun_uniform,
    'lecun_normal': lecun_normal,
}


def get_initializer(name):
    """
    根据名称获取初始化器

    Args:
        name: 初始化器名称

    Returns:
        初始化器函数

    Raises:
        ValueError: 如果初始化器不存在
    """
    if name not in INITIALIZERS:
        raise ValueError(f"未知的初始化器: {name}. 可用: {list(INITIALIZERS.keys())}")
    return INITIALIZERS[name]


def initialize_weights(shape, method='xavier_uniform', **kwargs):
    """
    便捷的权重初始化函数

    Args:
        shape: 权重形状
        method: 初始化方法名称
        **kwargs: 传递给初始化器的额外参数

    Returns:
        初始化后的权重

    Examples:
        >>> w = initialize_weights((10, 5), method='he_normal')
        >>> len(w), len(w[0])
        (10, 5)
    """
    initializer = get_initializer(method)
    return initializer(shape, **kwargs)
