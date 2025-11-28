# -*- coding: utf-8 -*-
"""
深度学习架构子包
"""

from .cnn import SimpleCNN, cnn_theory
from .rnn import SimpleRNN, rnn_theory
from .transformer import transformer_architecture

__all__ = [
    'SimpleCNN', 'cnn_theory',
    'SimpleRNN', 'rnn_theory',
    'transformer_architecture',
]
