# -*- coding: utf-8 -*-
"""
损失函数测试

测试 deep_learning/utils/losses.py 中的损失函数
"""

import sys
import os
import pytest
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deep_learning.utils import losses


@pytest.mark.unit
class TestMSELoss:
    """均方误差损失测试"""

    def test_mse_perfect_prediction(self):
        """测试完美预测的MSE为0"""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert losses.mse_loss(y_true, y_pred) == 0.0

    def test_mse_calculation(self):
        """测试MSE计算正确性"""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 2.9]
        # MSE = ((0.1)^2 + (0.1)^2 + (0.1)^2) / 3 = 0.01
        expected = 0.01
        assert abs(losses.mse_loss(y_true, y_pred) - expected) < 1e-6

    def test_mse_length_mismatch(self):
        """测试长度不匹配时抛出异常"""
        y_true = [1.0, 2.0]
        y_pred = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            losses.mse_loss(y_true, y_pred)


@pytest.mark.unit
class TestMAELoss:
    """平均绝对误差损失测试"""

    def test_mae_perfect_prediction(self):
        """测试完美预测的MAE为0"""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert losses.mae_loss(y_true, y_pred) == 0.0

    def test_mae_calculation(self):
        """测试MAE计算正确性"""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 2.9]
        # MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
        expected = 0.1
        assert abs(losses.mae_loss(y_true, y_pred) - expected) < 1e-6


@pytest.mark.unit
class TestBinaryCrossEntropy:
    """二分类交叉熵测试"""

    def test_bce_perfect_prediction(self):
        """测试完美预测的BCE接近0"""
        y_true = [1, 0, 1]
        y_pred = [0.9999, 0.0001, 0.9999]
        loss = losses.binary_cross_entropy(y_true, y_pred)
        assert loss < 0.001

    def test_bce_worst_prediction(self):
        """测试最差预测的BCE很大"""
        y_true = [1, 0]
        y_pred = [0.0001, 0.9999]
        loss = losses.binary_cross_entropy(y_true, y_pred)
        assert loss > 5.0  # 应该很大

    def test_bce_epsilon_prevents_log_zero(self):
        """测试epsilon防止log(0)"""
        y_true = [1, 0]
        y_pred = [1.0, 0.0]  # 极端值
        # 不应该抛出异常或返回inf
        loss = losses.binary_cross_entropy(y_true, y_pred)
        assert math.isfinite(loss)


@pytest.mark.unit
class TestCategoricalCrossEntropy:
    """多分类交叉熵测试"""

    def test_cce_perfect_prediction(self):
        """测试完美预测"""
        y_true = [1, 0, 0]  # one-hot for class 0
        y_pred = [0.9999, 0.0001, 0.0001]
        loss = losses.categorical_cross_entropy(y_true, y_pred)
        assert loss < 0.001

    def test_cce_calculation(self):
        """测试CCE计算"""
        y_true = [1, 0, 0]
        y_pred = [0.7, 0.2, 0.1]
        # -log(0.7) ≈ 0.357
        expected = -math.log(0.7)
        assert abs(losses.categorical_cross_entropy(y_true, y_pred) - expected) < 0.01

    def test_cce_epsilon_stability(self):
        """测试数值稳定性"""
        y_true = [1, 0]
        y_pred = [0.0, 1.0]  # 极端值
        loss = losses.categorical_cross_entropy(y_true, y_pred)
        assert math.isfinite(loss)


@pytest.mark.unit
class TestSparseCategoricalCrossEntropy:
    """稀疏多分类交叉熵测试"""

    def test_sparse_cce_single_sample(self):
        """测试单样本"""
        y_true_indices = [1]  # 类别1
        y_pred_probs = [[0.1, 0.7, 0.2]]  # 类别1的概率是0.7
        loss = losses.sparse_categorical_cross_entropy(y_true_indices, y_pred_probs)
        expected = -math.log(0.7)
        assert abs(loss - expected) < 0.01

    def test_sparse_cce_batch(self):
        """测试批次"""
        y_true_indices = [0, 2, 1]
        y_pred_probs = [
            [0.8, 0.1, 0.1],  # 样本0属于类别0
            [0.1, 0.2, 0.7],  # 样本1属于类别2
            [0.2, 0.6, 0.2],  # 样本2属于类别1
        ]
        loss = losses.sparse_categorical_cross_entropy(y_true_indices, y_pred_probs)
        # 应该是 (-log(0.8) - log(0.7) - log(0.6)) / 3
        expected = (-math.log(0.8) - math.log(0.7) - math.log(0.6)) / 3
        assert abs(loss - expected) < 0.01


@pytest.mark.unit
class TestHingeLoss:
    """Hinge损失测试"""

    def test_hinge_correct_classification(self):
        """测试正确分类（间隔>1）"""
        y_true = [1, -1]
        y_pred = [2.0, -2.0]  # 间隔为2
        loss = losses.hinge_loss(y_true, y_pred)
        assert loss == 0.0  # max(0, 1-2) + max(0, 1-2) = 0

    def test_hinge_incorrect_classification(self):
        """测试错误分类"""
        y_true = [1, -1]
        y_pred = [-1.0, 1.0]  # 完全错误
        loss = losses.hinge_loss(y_true, y_pred)
        # max(0, 1-(-1)) + max(0, 1-(-1)) = 2 + 2 = 4, 平均2
        assert loss == 2.0

    def test_hinge_margin(self):
        """测试间隔效果"""
        y_true = [1]
        y_pred = [0.8]
        loss = losses.hinge_loss(y_true, y_pred)
        # max(0, 1 - 0.8) = 0.2
        assert abs(loss - 0.2) < 1e-6


@pytest.mark.unit
class TestHuberLoss:
    """Huber损失测试"""

    def test_huber_small_error(self):
        """测试小误差（使用二次形式）"""
        y_true = [1.0]
        y_pred = [1.1]  # 误差0.1 < delta
        loss = losses.huber_loss(y_true, y_pred, delta=1.0)
        # 0.5 * 0.1^2 = 0.005
        assert abs(loss - 0.005) < 1e-6

    def test_huber_large_error(self):
        """测试大误差（使用线性形式）"""
        y_true = [1.0]
        y_pred = [3.0]  # 误差2.0 > delta(1.0)
        loss = losses.huber_loss(y_true, y_pred, delta=1.0)
        # delta * (|error| - 0.5*delta) = 1.0 * (2.0 - 0.5) = 1.5
        assert abs(loss - 1.5) < 1e-6

    def test_huber_perfect_prediction(self):
        """测试完美预测"""
        y_true = [1.0, 2.0]
        y_pred = [1.0, 2.0]
        loss = losses.huber_loss(y_true, y_pred)
        assert loss == 0.0


@pytest.mark.unit
class TestLossRegistry:
    """损失函数注册表测试"""

    def test_get_loss_function(self):
        """测试根据名称获取损失函数"""
        mse_fn = losses.get_loss_function('mse')
        assert mse_fn([1, 2], [1, 2]) == losses.mse_loss([1, 2], [1, 2])

    def test_get_loss_function_invalid(self):
        """测试获取不存在的损失函数"""
        with pytest.raises(ValueError):
            losses.get_loss_function('invalid_loss')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
