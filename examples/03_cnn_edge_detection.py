# -*- coding: utf-8 -*-
"""
示例 03: CNN 边缘检测（简化版）
"""

from deep_learning.architectures import SimpleCNN


def main():
    cnn = SimpleCNN(
        input_shape=(5, 5, 1),
        conv_filters=[(1, 3, 1)],
        fc_layers=[4],
    )

    # 简单垂直边缘
    image = [
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
    kernel = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]

    feature_map = cnn.conv2d(image, kernel, bias=0, stride=1)
    print("Edge map (中心值):", feature_map[1][1])


if __name__ == "__main__":
    main()
