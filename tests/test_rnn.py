# -*- coding: utf-8 -*-
"""
RNN 模块单元测试

测试 deep_learning_rnn.py 中的核心功能
"""

import sys
import os
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用新包路径
from deep_learning.architectures import SimpleRNN


@pytest.mark.unit
class TestRNNInitialization:
    """RNN 初始化测试"""

    def test_rnn_initialization(self):
        """测试 RNN 初始化"""
        rnn = SimpleRNN(
            input_size=3,
            hidden_size=5,
            output_size=2,
            learning_rate=0.01
        )

        assert rnn.input_size == 3
        assert rnn.hidden_size == 5
        assert rnn.output_size == 2
        assert rnn.learning_rate == 0.01

    def test_rnn_weights_initialization(self):
        """测试权重矩阵初始化"""
        rnn = SimpleRNN(
            input_size=4,
            hidden_size=6,
            output_size=3
        )

        # Wxh: input_size x hidden_size
        assert len(rnn.Wxh) == 4
        assert len(rnn.Wxh[0]) == 6

        # Whh: hidden_size x hidden_size
        assert len(rnn.Whh) == 6
        assert len(rnn.Whh[0]) == 6

        # Why: hidden_size x output_size
        assert len(rnn.Why) == 6
        assert len(rnn.Why[0]) == 3

        # 偏置
        assert len(rnn.bh) == 6
        assert len(rnn.by) == 3

    def test_parameter_count(self):
        """测试参数数量计算"""
        rnn = SimpleRNN(input_size=3, hidden_size=4, output_size=2)

        param_count = rnn.count_parameters()

        # Wxh: 3*4 = 12
        # Whh: 4*4 = 16
        # Why: 4*2 = 8
        # bh: 4
        # by: 2
        # Total: 12 + 16 + 8 + 4 + 2 = 42
        assert param_count == 42


@pytest.mark.unit
class TestRNNActivations:
    """RNN 激活函数测试"""

    def test_tanh_activation(self):
        """测试 tanh 激活函数"""
        rnn = SimpleRNN(input_size=1, hidden_size=1, output_size=1)

        # tanh(0) = 0
        assert abs(rnn.tanh(0)) < 1e-6

        # tanh(大值) ≈ 1
        assert abs(rnn.tanh(10) - 1.0) < 0.01

        # tanh(小值) ≈ -1
        assert abs(rnn.tanh(-10) - (-1.0)) < 0.01

        # tanh 是奇函数
        x = 0.5
        assert abs(rnn.tanh(x) + rnn.tanh(-x)) < 1e-6

    def test_softmax_activation(self):
        """测试 softmax 激活函数"""
        rnn = SimpleRNN(input_size=1, hidden_size=1, output_size=3)

        logits = [1.0, 2.0, 3.0]
        probs = rnn.softmax(logits)

        # 概率之和应该为 1
        assert abs(sum(probs) - 1.0) < 1e-6

        # 所有概率应该在 [0, 1]
        assert all(0 <= p <= 1 for p in probs)

        # 最大 logit 应该对应最大概率
        assert probs.index(max(probs)) == logits.index(max(logits))


@pytest.mark.unit
def test_matrix_vector_multiply():
    """测试矩阵向量乘法"""
    rnn = SimpleRNN(input_size=1, hidden_size=1, output_size=1)

    # 2x3 矩阵
    matrix = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    # 3维向量
    vector = [1, 0, -1]

    result = rnn.matrix_vector_multiply(matrix, vector)

    # 结果应该是 2维
    assert len(result) == 2

    # 验证计算
    # [1,2,3] · [1,0,-1] = 1 - 3 = -2
    # [4,5,6] · [1,0,-1] = 4 - 6 = -2
    assert result[0] == -2
    assert result[1] == -2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
