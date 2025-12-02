# -*- coding: utf-8 -*-
"""
NumPy 后端功能测试
"""

import pytest

from deep_learning.utils import (
    conv2d_single_channel,
    matrix_multiply,
    numpy_available,
    numpy_conv2d_single_channel,
    numpy_matmul,
)


@pytest.mark.unit
def test_numpy_available():
    assert numpy_available() is True


@pytest.mark.unit
def test_numpy_matmul_matches_python():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = matrix_multiply(A, B)
    result = numpy_matmul(A, B).tolist()
    assert result == expected


@pytest.mark.unit
def test_numpy_conv2d_matches_python():
    input_map = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    kernel = [
        [1, 0],
        [0, -1],
    ]
    bias = 0.5

    expected = conv2d_single_channel(input_map, kernel, bias=bias, stride=1)
    result = numpy_conv2d_single_channel(input_map, kernel, bias=bias, stride=1).tolist()
    assert result == expected
