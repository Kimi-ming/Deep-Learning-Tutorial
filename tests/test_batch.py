# -*- coding: utf-8 -*-
"""
批处理前向的基本测试
"""

from deep_learning.fundamentals import MLP
from deep_learning.architectures import SimpleCNN, SimpleRNN
from deep_learning.architectures.cnn import forward_batch as cnn_forward_batch


def test_mlp_forward_batch():
    mlp = MLP(layers=[2, 2], learning_rate=0.1)
    batch = [[0, 0], [1, 1]]
    outs = mlp.forward_batch(batch, training=False)
    assert len(outs) == len(batch)


def test_cnn_forward_batch():
    cnn = SimpleCNN((3, 3, 1), [(1, 2, 1)], [2])
    batch = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    outs = cnn_forward_batch(cnn, batch)
    assert len(outs) == len(batch)


def test_rnn_forward_batch():
    rnn = SimpleRNN(input_size=2, hidden_size=2, output_size=1)
    batch = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
    ]
    outs = rnn.forward_batch(batch)
    assert len(outs) == len(batch)
