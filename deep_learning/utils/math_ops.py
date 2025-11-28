# -*- coding: utf-8 -*-
"""
数学运算工具模块

提供神经网络中常用的数学运算。

功能:
- 矩阵/向量运算
- 梯度裁剪
- 归一化
- 距离度量
"""

import math


def matrix_vector_multiply(matrix, vector):
    """
    矩阵向量乘法: M @ v

    Args:
        matrix: 矩阵，形状 (m, n)
        vector: 向量，长度 n

    Returns:
        结果向量，长度 m

    Examples:
        >>> M = [[1, 2], [3, 4], [5, 6]]
        >>> v = [1, 2]
        >>> result = matrix_vector_multiply(M, v)
        >>> result
        [5, 11, 17]
    """
    if not matrix or not vector:
        raise ValueError("矩阵和向量不能为空")

    n_cols = len(matrix[0])
    if len(vector) != n_cols:
        raise ValueError(f"维度不匹配: matrix有{n_cols}列, vector长度为{len(vector)}")

    result = []
    for row in matrix:
        dot_product = sum(m * v for m, v in zip(row, vector))
        result.append(dot_product)

    return result


def matrix_multiply(A, B):
    """
    矩阵乘法: A @ B

    Args:
        A: 左矩阵，形状 (m, n)
        B: 右矩阵，形状 (n, p)

    Returns:
        结果矩阵，形状 (m, p)

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> C = matrix_multiply(A, B)
        >>> C
        [[19, 22], [43, 50]]
    """
    if not A or not B:
        raise ValueError("矩阵不能为空")

    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])

    if n != n2:
        raise ValueError(f"维度不匹配: A的列数{n} != B的行数{n2}")

    # 将 B 转置，减少索引层数，加速纯 Python 循环
    BT = list(zip(*B))
    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in BT] for row_a in A]


def transpose(matrix):
    """
    矩阵转置

    Args:
        matrix: 输入矩阵，形状 (m, n)

    Returns:
        转置矩阵，形状 (n, m)

    Examples:
        >>> M = [[1, 2, 3], [4, 5, 6]]
        >>> transpose(M)
        [[1, 4], [2, 5], [3, 6]]
    """
    if not matrix:
        return []

    m, n = len(matrix), len(matrix[0])
    result = [[matrix[i][j] for i in range(m)] for j in range(n)]

    return result


def vector_add(a, b):
    """
    向量加法: a + b

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        和向量

    Examples:
        >>> vector_add([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
    """
    if len(a) != len(b):
        raise ValueError(f"向量长度不匹配: {len(a)} vs {len(b)}")

    return [x + y for x, y in zip(a, b)]


def vector_subtract(a, b):
    """
    向量减法: a - b

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        差向量
    """
    if len(a) != len(b):
        raise ValueError(f"向量长度不匹配: {len(a)} vs {len(b)}")

    return [x - y for x, y in zip(a, b)]


def vector_scalar_multiply(vector, scalar):
    """
    向量数乘: scalar * vector

    Args:
        vector: 向量
        scalar: 标量

    Returns:
        缩放后的向量

    Examples:
        >>> vector_scalar_multiply([1, 2, 3], 2)
        [2, 4, 6]
    """
    return [scalar * x for x in vector]


def dot_product(a, b):
    """
    向量点积: a · b

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        点积值（标量）

    Examples:
        >>> dot_product([1, 2, 3], [4, 5, 6])
        32
    """
    if len(a) != len(b):
        raise ValueError(f"向量长度不匹配: {len(a)} vs {len(b)}")

    return sum(x * y for x, y in zip(a, b))


def vector_norm(vector, p=2):
    """
    向量范数

    Args:
        vector: 输入向量
        p: 范数类型（1: L1范数, 2: L2范数, float('inf'): 无穷范数）

    Returns:
        范数值

    Examples:
        >>> vector_norm([3, 4])  # L2范数
        5.0
        >>> vector_norm([3, 4], p=1)  # L1范数
        7.0
    """
    if p == 1:
        # L1范数: Σ|x_i|
        return sum(abs(x) for x in vector)
    elif p == 2:
        # L2范数: sqrt(Σx_i²)
        return math.sqrt(sum(x * x for x in vector))
    elif p == float('inf'):
        # 无穷范数: max|x_i|
        return max(abs(x) for x in vector)
    else:
        # Lp范数: (Σ|x_i|^p)^(1/p)
        return sum(abs(x) ** p for x in vector) ** (1.0 / p)


def clip_gradients(gradients, max_norm):
    """
    梯度裁剪（防止梯度爆炸）

    如果梯度范数超过max_norm，则缩放到max_norm

    Args:
        gradients: 梯度向量或梯度列表
        max_norm: 最大范数阈值

    Returns:
        裁剪后的梯度

    Examples:
        >>> grads = [3.0, 4.0]  # 范数为5
        >>> clipped = clip_gradients(grads, max_norm=2.5)
        >>> vector_norm(clipped)  # 应该接近2.5
        2.5
    """
    # 处理嵌套列表（多层梯度）
    if isinstance(gradients[0], list):
        # 扁平化
        flat_grads = []
        for grad_matrix in gradients:
            for grad_row in grad_matrix:
                if isinstance(grad_row, list):
                    flat_grads.extend(grad_row)
                else:
                    flat_grads.append(grad_row)

        # 计算全局范数
        total_norm = vector_norm(flat_grads, p=2)

        # 计算缩放因子
        if total_norm > max_norm:
            scale = max_norm / total_norm

            # 缩放所有梯度
            clipped_gradients = []
            for grad_matrix in gradients:
                if isinstance(grad_matrix[0], list):
                    # 矩阵梯度
                    clipped_matrix = [[g * scale for g in row] for row in grad_matrix]
                    clipped_gradients.append(clipped_matrix)
                else:
                    # 向量梯度
                    clipped_gradients.append([g * scale for g in grad_matrix])

            return clipped_gradients
        else:
            return gradients
    else:
        # 简单向量
        norm = vector_norm(gradients, p=2)

        if norm > max_norm:
            scale = max_norm / norm
            return [g * scale for g in gradients]
        else:
            return gradients


def clip_by_value(values, min_value, max_value):
    """
    按值裁剪（限制范围）

    Args:
        values: 值列表或嵌套列表
        min_value: 最小值
        max_value: 最大值

    Returns:
        裁剪后的值

    Examples:
        >>> clip_by_value([0.5, 1.5, 2.5], 1.0, 2.0)
        [1.0, 1.5, 2.0]
    """
    if isinstance(values[0], list):
        # 嵌套列表
        return [[max(min_value, min(max_value, v)) for v in row] for row in values]
    else:
        # 简单列表
        return [max(min_value, min(max_value, v)) for v in values]


def normalize_vector(vector, epsilon=1e-8):
    """
    向量归一化（单位化）

    Args:
        vector: 输入向量
        epsilon: 避免除以0

    Returns:
        归一化后的向量

    Examples:
        >>> v = normalize_vector([3, 4])
        >>> abs(vector_norm(v) - 1.0) < 1e-6
        True
    """
    norm = vector_norm(vector, p=2)

    if norm < epsilon:
        # 接近零向量，返回原向量
        return vector

    return [x / norm for x in vector]


def batch_normalize(batch, epsilon=1e-8):
    """
    批归一化（减均值除标准差）

    Args:
        batch: 样本列表，每个样本是一个向量
        epsilon: 数值稳定性参数

    Returns:
        归一化后的批次

    Examples:
        >>> batch = [[1, 2], [3, 4], [5, 6]]
        >>> normalized = batch_normalize(batch)
        >>> # 每个特征的均值接近0
    """
    if not batch:
        return batch

    batch_size = len(batch)
    feature_dim = len(batch[0])

    # 计算每个特征的均值
    means = [0.0] * feature_dim
    for sample in batch:
        for i, val in enumerate(sample):
            means[i] += val
    means = [m / batch_size for m in means]

    # 计算每个特征的方差
    variances = [0.0] * feature_dim
    for sample in batch:
        for i, val in enumerate(sample):
            variances[i] += (val - means[i]) ** 2
    variances = [v / batch_size for v in variances]

    # 归一化
    normalized_batch = []
    for sample in batch:
        normalized_sample = []
        for i, val in enumerate(sample):
            normalized_val = (val - means[i]) / math.sqrt(variances[i] + epsilon)
            normalized_sample.append(normalized_val)
        normalized_batch.append(normalized_sample)

    return normalized_batch


def cosine_similarity(a, b):
    """
    余弦相似度

    数学公式: cos(θ) = (a · b) / (||a|| * ||b||)

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        余弦相似度，范围 [-1, 1]

    Examples:
        >>> cosine_similarity([1, 0], [1, 0])  # 相同方向
        1.0
        >>> cosine_similarity([1, 0], [-1, 0])  # 相反方向
        -1.0
    """
    if len(a) != len(b):
        raise ValueError(f"向量长度不匹配: {len(a)} vs {len(b)}")

    dot = dot_product(a, b)
    norm_a = vector_norm(a, p=2)
    norm_b = vector_norm(b, p=2)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return dot / (norm_a * norm_b)


def euclidean_distance(a, b):
    """
    欧几里得距离

    数学公式: d = sqrt(Σ(a_i - b_i)²)

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        欧几里得距离

    Examples:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    if len(a) != len(b):
        raise ValueError(f"向量长度不匹配: {len(a)} vs {len(b)}")

    squared_diff = [(x - y) ** 2 for x, y in zip(a, b)]
    return math.sqrt(sum(squared_diff))


def moving_average(values, window_size):
    """
    移动平均（平滑时间序列）

    Args:
        values: 值序列
        window_size: 窗口大小

    Returns:
        平滑后的序列

    Examples:
        >>> moving_average([1, 2, 3, 4, 5], window_size=3)
        [2.0, 3.0, 4.0]
    """
    if window_size > len(values):
        raise ValueError(f"窗口大小{window_size}超过序列长度{len(values)}")

    smoothed = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        smoothed.append(sum(window) / window_size)

    return smoothed
