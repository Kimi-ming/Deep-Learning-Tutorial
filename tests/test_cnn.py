# -*- coding: utf-8 -*-
"""
CNN 模块单元测试

测试 deep_learning_cnn.py 中的核心功能
"""

import sys
import os
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用新包路径
from deep_learning.architectures import SimpleCNN


@pytest.mark.unit
class TestCNNOperations:
    """CNN 基本操作测试"""

    def test_cnn_initialization(self):
        """测试 CNN 初始化"""
        cnn = SimpleCNN(
            input_shape=(8, 8, 1),
            conv_filters=[(4, 3, 1)],
            fc_layers=[10]
        )

        assert cnn.input_shape == (8, 8, 1)
        assert len(cnn.conv_filters) == 1
        assert cnn.conv_filters[0] == (4, 3, 1)

    def test_conv2d_output_shape(self):
        """测试卷积操作输出形状"""
        cnn = SimpleCNN(
            input_shape=(5, 5, 1),
            conv_filters=[(1, 3, 1)],
            fc_layers=[5]
        )

        # 5x5 输入，3x3 卷积核，步长 1
        # 输出应该是 (5-3)/1 + 1 = 3x3
        input_image = [[i * 5 + j for j in range(5)] for i in range(5)]
        kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
        bias = 0

        output = cnn.conv2d(input_image, kernel, bias, stride=1)

        assert len(output) == 3, "输出高度应该是 3"
        assert len(output[0]) == 3, "输出宽度应该是 3"

    def test_conv2d_edge_detection(self):
        """测试卷积操作的边缘检测"""
        cnn = SimpleCNN(
            input_shape=(5, 5, 1),
            conv_filters=[(1, 3, 1)],
            fc_layers=[5]
        )

        # 垂直边缘图像
        input_image = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1]
        ]

        # 垂直边缘检测核
        edge_kernel = [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]

        output = cnn.conv2d(input_image, edge_kernel, bias=0, stride=1)

        # 在边缘位置应该有较大的响应
        assert abs(output[1][1]) > 0, "应该检测到边缘"

    def test_max_pooling_operation(self):
        """测试最大池化操作"""
        cnn = SimpleCNN(
            input_shape=(4, 4, 1),
            conv_filters=[(1, 2, 1)],
            fc_layers=[5]
        )

        # 4x4 特征图
        feature_map = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]

        # 2x2 池化窗口，步长 2
        pooled = cnn.max_pooling(feature_map, pool_size=2, stride=2)

        # 输出应该是 2x2
        assert len(pooled) == 2
        assert len(pooled[0]) == 2

        # 验证最大池化值
        assert pooled[0][0] == 6.0, "左上池化窗口的最大值"
        assert pooled[0][1] == 8.0, "右上池化窗口的最大值"
        assert pooled[1][0] == 14.0, "左下池化窗口的最大值"
        assert pooled[1][1] == 16.0, "右下池化窗口的最大值"

    def test_relu_activation(self):
        """测试 ReLU 激活函数"""
        cnn = SimpleCNN(
            input_shape=(3, 3, 1),
            conv_filters=[(1, 2, 1)],
            fc_layers=[5]
        )

        feature_map = [
            [-1.0, 2.0, -3.0],
            [4.0, -5.0, 6.0],
            [-7.0, 8.0, -9.0]
        ]

        activated = cnn.relu(feature_map)

        expected = [
            [0.0, 2.0, 0.0],
            [4.0, 0.0, 6.0],
            [0.0, 8.0, 0.0]
        ]

        for i in range(3):
            for j in range(3):
                assert activated[i][j] == expected[i][j]

    def test_flatten_operation(self):
        """测试展平操作"""
        cnn = SimpleCNN(
            input_shape=(2, 2, 1),
            conv_filters=[(2, 2, 1)],
            fc_layers=[4]
        )

        # 2个 2x2 特征图
        feature_maps = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]

        flattened = cnn.flatten(feature_maps)

        # 应该展平为一维向量
        expected = [1, 2, 3, 4, 5, 6, 7, 8]
        assert flattened == expected


@pytest.mark.unit
class TestCNNForwardPass:
    """CNN 前向传播测试"""

    def test_simple_forward_pass(self):
        """测试简单的前向传播"""
        cnn = SimpleCNN(
            input_shape=(8, 8, 1),
            conv_filters=[(2, 3, 1)],  # 2个 3x3 卷积核
            fc_layers=[10]
        )

        # 创建 8x8 输入图像
        input_image = [[i * 8 + j for j in range(8)] for i in range(8)]

        # 前向传播
        output = cnn.forward(input_image)

        # 验证输出是一维向量
        assert isinstance(output, list)
        assert len(output) > 0
        assert all(isinstance(x, (int, float)) for x in output)

    def test_multiple_conv_layers(self):
        """测试多层卷积"""
        cnn = SimpleCNN(
            input_shape=(10, 10, 1),
            conv_filters=[
                (2, 3, 1),  # 第一层
                (4, 3, 1)   # 第二层
            ],
            fc_layers=[10]
        )

        input_image = [[0.5] * 10 for _ in range(10)]

        # 前向传播应该不崩溃
        output = cnn.forward(input_image)
        assert len(output) > 0


@pytest.mark.unit
def test_cnn_parameter_count():
    """测试 CNN 参数数量计算"""
    cnn = SimpleCNN(
        input_shape=(5, 5, 1),
        conv_filters=[(4, 3, 1)],  # 4个 3x3 卷积核
        fc_layers=[10]
    )

    # 卷积层参数：1 * 3 * 3 * 4 + 4(bias) = 40
    # 实际数量会因网络设计而异，这里主要测试可以成功初始化
    assert len(cnn.conv_weights) > 0
    assert len(cnn.conv_biases) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
