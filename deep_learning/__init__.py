# -*- coding: utf-8 -*-
"""
Deep Learning Tutorial Package

纯Python实现的深度学习教学框架，无需外部深度学习库。

子包:
- utils: 工具函数（激活函数、损失函数、初始化器、数学运算）
- fundamentals: 基础模块（感知机、MLP等）
- architectures: 网络架构（CNN、RNN、Transformer等）
- optimizers: 优化器（SGD、Adam等）
- advanced: 高级模块（GAN、VAE、NAS等）
"""

__version__ = '0.1.0'
__author__ = 'Deep Learning Tutorial Contributors'

# 导入utils包，方便使用
from . import utils

__all__ = ['utils']
