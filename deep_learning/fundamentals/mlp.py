# -*- coding: utf-8 -*-
"""
多层感知机包装

迁移版：直接使用新包中的 DeepNetwork。
"""

from .deep_network import DeepNetwork


class MLP(DeepNetwork):
    """多层感知机，继承自 DeepNetwork"""

    def __init__(self, layers, learning_rate: float = 0.001, activation: str = 'relu',
                 use_batch_norm: bool = False, dropout_rate: float = 0.0):
        super().__init__(
            layers=layers,
            learning_rate=learning_rate,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )


__all__ = ['MLP', 'DeepNetwork']
