# -*- coding: utf-8 -*-
"""
数学运算关键路径的输出校验（确保优化后结果一致）
"""

import pytest

from deep_learning.utils import conv2d_single_channel, matrix_multiply


@pytest.mark.unit
def test_matrix_multiply_output_matches_expected():
    A = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    B = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    expected = [
        [58, 64],
        [139, 154],
    ]

    assert matrix_multiply(A, B) == expected


@pytest.mark.unit
def test_conv2d_single_channel_stride_and_bias():
    input_map = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    # 斜对角核，验证 stride>1 与 bias 计算
    kernel = [
        [1, 0],
        [0, 1],
    ]

    result = conv2d_single_channel(input_map, kernel, bias=1.0, stride=2)
    expected = [
        [8.0, 12.0],
        [24.0, 28.0],
    ]

    assert result == expected
