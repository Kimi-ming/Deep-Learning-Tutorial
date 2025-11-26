# -*- coding: utf-8 -*-
"""
深度学习基础模块测试

测试 deep_learning_fundamentals.py 的基本功能
"""

import sys
import os
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deep_learning_fundamentals


@pytest.mark.unit
class TestModuleImport:
    """模块导入测试"""

    def test_module_imports(self):
        """测试模块可以成功导入"""
        assert deep_learning_fundamentals is not None

    def test_deep_network_class_exists(self):
        """测试 DeepNetwork 类存在"""
        assert hasattr(deep_learning_fundamentals, 'DeepNetwork')


@pytest.mark.unit
class TestDeepNetwork:
    """DeepNetwork 类基本测试"""

    def test_network_creation(self):
        """测试网络可以创建"""
        network = deep_learning_fundamentals.DeepNetwork(
            layers=[2, 3, 1],
            learning_rate=0.01
        )

        assert network is not None
        assert network.layers == [2, 3, 1]
        assert network.learning_rate == 0.01

    def test_network_has_weights_and_biases(self):
        """测试网络有权重和偏置"""
        network = deep_learning_fundamentals.DeepNetwork(layers=[2, 3, 1])

        # 应该有 2 层权重（输入->隐藏, 隐藏->输出）
        assert hasattr(network, 'weights')
        assert hasattr(network, 'biases')
        assert len(network.weights) == 2
        assert len(network.biases) == 2

    def test_network_forward_method_exists(self):
        """测试网络有 forward 方法"""
        network = deep_learning_fundamentals.DeepNetwork(layers=[2, 2])

        assert hasattr(network, 'forward')
        assert callable(network.forward)


@pytest.mark.unit
class TestModuleFunctions:
    """模块函数测试"""

    def test_introduction_function_exists(self):
        """测试介绍函数存在"""
        assert hasattr(deep_learning_fundamentals, 'deep_learning_introduction')
        assert callable(deep_learning_fundamentals.deep_learning_introduction)

    def test_architectures_overview_exists(self):
        """测试架构概览函数存在"""
        assert hasattr(deep_learning_fundamentals, 'deep_learning_architectures_overview')
        assert callable(deep_learning_fundamentals.deep_learning_architectures_overview)

    def test_main_function_exists(self):
        """测试 main 函数存在"""
        assert hasattr(deep_learning_fundamentals, 'main')
        assert callable(deep_learning_fundamentals.main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
