# -*- coding: utf-8 -*-
"""
Deep Learning Utils Package

提供神经网络训练所需的各种工具函数。

子模块:
- activations: 激活函数
- losses: 损失函数
- initializers: 权重初始化
- math_ops: 数学运算
"""

# 导入常用函数，方便使用
from .activations import (
    sigmoid, tanh, relu, leaky_relu, softmax,
    sigmoid_derivative, tanh_derivative, relu_derivative,
    get_activation, get_activation_derivative
)

from .losses import (
    mse_loss, mae_loss,
    binary_cross_entropy, categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    hinge_loss, huber_loss,
    get_loss_function
)

from .initializers import (
    zeros, ones, uniform, normal,
    xavier_uniform, xavier_normal,
    he_uniform, he_normal,
    lecun_uniform, lecun_normal,
    get_initializer, initialize_weights
)

from .math_ops import (
    matrix_vector_multiply, matrix_multiply, transpose,
    vector_add, vector_subtract, vector_scalar_multiply,
    dot_product, vector_norm,
    clip_gradients, clip_by_value,
    normalize_vector, batch_normalize,
    cosine_similarity, euclidean_distance,
    moving_average, conv2d_single_channel
)
from .visualization import plot_loss_curve, plot_accuracy_curve
from .performance import EarlyStopping, GradientAccumulator

__all__ = [
    # Activations
    'sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax',
    'sigmoid_derivative', 'tanh_derivative', 'relu_derivative',
    'get_activation', 'get_activation_derivative',
    # Losses
    'mse_loss', 'mae_loss',
    'binary_cross_entropy', 'categorical_cross_entropy',
    'sparse_categorical_cross_entropy',
    'hinge_loss', 'huber_loss',
    'get_loss_function',
    # Initializers
    'zeros', 'ones', 'uniform', 'normal',
    'xavier_uniform', 'xavier_normal',
    'he_uniform', 'he_normal',
    'lecun_uniform', 'lecun_normal',
    'get_initializer', 'initialize_weights',
    # Math Operations
    'matrix_vector_multiply', 'matrix_multiply', 'transpose',
    'vector_add', 'vector_subtract', 'vector_scalar_multiply',
    'dot_product', 'vector_norm',
    'clip_gradients', 'clip_by_value',
    'normalize_vector', 'batch_normalize',
    'cosine_similarity', 'euclidean_distance',
    'moving_average', 'conv2d_single_channel',
    # Visualization
    'plot_loss_curve', 'plot_accuracy_curve',
    # Performance
    'EarlyStopping', 'GradientAccumulator',
]
