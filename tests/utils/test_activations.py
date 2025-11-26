# -*- coding: utf-8 -*-
"""
激活函数测试

测试 deep_learning/utils/activations.py 中的激活函数
"""

import sys
import os
import pytest
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deep_learning.utils import activations


@pytest.mark.unit
class TestSigmoid:
    """Sigmoid激活函数测试"""

    def test_sigmoid_zero(self):
        """测试sigmoid(0) = 0.5"""
        assert abs(activations.sigmoid(0) - 0.5) < 1e-6

    def test_sigmoid_large_positive(self):
        """测试大正数时sigmoid接近1"""
        assert activations.sigmoid(100) > 0.99
        assert activations.sigmoid(500) == 1.0  # 数值稳定性

    def test_sigmoid_large_negative(self):
        """测试大负数时sigmoid接近0"""
        assert activations.sigmoid(-100) < 0.01
        assert activations.sigmoid(-500) < 1e-200  # 数值稳定性 - 极小值

    def test_sigmoid_symmetry(self):
        """测试sigmoid的对称性: σ(-x) = 1 - σ(x)"""
        x = 2.0
        assert abs(activations.sigmoid(-x) - (1 - activations.sigmoid(x))) < 1e-6

    def test_sigmoid_range(self):
        """测试sigmoid输出范围"""
        for x in [-10, -5, 0, 5, 10]:
            result = activations.sigmoid(x)
            assert 0 < result < 1


@pytest.mark.unit
class TestTanh:
    """Tanh激活函数测试"""

    def test_tanh_zero(self):
        """测试tanh(0) = 0"""
        assert abs(activations.tanh(0)) < 1e-6

    def test_tanh_large_positive(self):
        """测试大正数时tanh接近1"""
        assert abs(activations.tanh(10) - 1.0) < 0.01
        assert activations.tanh(20) == 1.0  # 数值稳定性

    def test_tanh_large_negative(self):
        """测试大负数时tanh接近-1"""
        assert abs(activations.tanh(-10) - (-1.0)) < 0.01
        assert activations.tanh(-20) == -1.0  # 数值稳定性

    def test_tanh_odd_function(self):
        """测试tanh是奇函数: tanh(-x) = -tanh(x)"""
        x = 1.5
        assert abs(activations.tanh(-x) + activations.tanh(x)) < 1e-6

    def test_tanh_range(self):
        """测试tanh输出范围"""
        for x in [-10, -5, 0, 5, 10]:
            result = activations.tanh(x)
            assert -1 <= result <= 1


@pytest.mark.unit
class TestReLU:
    """ReLU激活函数测试"""

    def test_relu_positive(self):
        """测试ReLU对正数保持不变"""
        assert activations.relu(5.0) == 5.0
        assert activations.relu(0.1) == 0.1

    def test_relu_negative(self):
        """测试ReLU将负数变为0"""
        assert activations.relu(-5.0) == 0.0
        assert activations.relu(-0.1) == 0.0

    def test_relu_zero(self):
        """测试ReLU(0) = 0"""
        assert activations.relu(0.0) == 0.0


@pytest.mark.unit
class TestLeakyReLU:
    """Leaky ReLU激活函数测试"""

    def test_leaky_relu_positive(self):
        """测试对正数保持不变"""
        assert activations.leaky_relu(5.0) == 5.0

    def test_leaky_relu_negative_default(self):
        """测试对负数使用默认alpha"""
        assert activations.leaky_relu(-10.0) == -0.1  # alpha=0.01

    def test_leaky_relu_negative_custom(self):
        """测试对负数使用自定义alpha"""
        assert activations.leaky_relu(-10.0, alpha=0.2) == -2.0

    def test_leaky_relu_zero(self):
        """测试Leaky ReLU(0) = 0"""
        assert activations.leaky_relu(0.0) == 0.0


@pytest.mark.unit
class TestSoftmax:
    """Softmax激活函数测试"""

    def test_softmax_sum_to_one(self):
        """测试softmax输出和为1"""
        logits = [1.0, 2.0, 3.0]
        probs = activations.softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_softmax_range(self):
        """测试softmax输出在[0,1]范围"""
        logits = [1.0, 2.0, 3.0]
        probs = activations.softmax(logits)
        assert all(0 <= p <= 1 for p in probs)

    def test_softmax_max_logit(self):
        """测试最大logit对应最大概率"""
        logits = [1.0, 3.0, 2.0]
        probs = activations.softmax(logits)
        assert probs.index(max(probs)) == 1  # 索引1的logit最大

    def test_softmax_numerical_stability(self):
        """测试数值稳定性（大数值）"""
        logits = [1000.0, 1001.0, 1002.0]
        probs = activations.softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(0 <= p <= 1 for p in probs)

    def test_softmax_uniform(self):
        """测试相同logit产生均匀分布"""
        logits = [1.0, 1.0, 1.0]
        probs = activations.softmax(logits)
        expected = 1.0 / 3
        for p in probs:
            assert abs(p - expected) < 1e-6


@pytest.mark.unit
class TestActivationDerivatives:
    """激活函数导数测试"""

    def test_sigmoid_derivative(self):
        """测试sigmoid导数"""
        # σ'(0) = 0.25
        assert abs(activations.sigmoid_derivative(0) - 0.25) < 0.01

    def test_tanh_derivative(self):
        """测试tanh导数"""
        # tanh'(0) = 1
        assert abs(activations.tanh_derivative(0) - 1.0) < 0.01

    def test_relu_derivative(self):
        """测试ReLU导数"""
        assert activations.relu_derivative(5.0) == 1.0
        assert activations.relu_derivative(-5.0) == 0.0
        assert activations.relu_derivative(0.0) == 0.0


@pytest.mark.unit
class TestActivationRegistry:
    """激活函数注册表测试"""

    def test_get_activation(self):
        """测试根据名称获取激活函数"""
        sigmoid_fn = activations.get_activation('sigmoid')
        assert sigmoid_fn(0) == activations.sigmoid(0)

        relu_fn = activations.get_activation('relu')
        assert relu_fn(5) == activations.relu(5)

    def test_get_activation_invalid(self):
        """测试获取不存在的激活函数"""
        with pytest.raises(ValueError):
            activations.get_activation('invalid_activation')

    def test_get_activation_derivative(self):
        """测试根据名称获取激活函数导数"""
        sigmoid_derivative_fn = activations.get_activation_derivative('sigmoid')
        assert sigmoid_derivative_fn(0) == activations.sigmoid_derivative(0)

    def test_get_activation_derivative_invalid(self):
        """测试获取不存在的导数"""
        with pytest.raises(ValueError):
            activations.get_activation_derivative('invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
