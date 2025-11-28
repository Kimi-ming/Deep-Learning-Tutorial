# -*- coding: utf-8 -*-
"""
DataLoader 测试
"""

from datasets.data_loader import DataLoader


def test_load_mnist_sample_shapes():
    loader = DataLoader()
    images, labels = loader.load_mnist_sample()
    assert images.shape[1:] == (28, 28)
    assert len(images) == len(labels)
    assert images.dtype == "uint8"


def test_load_text_sequences():
    loader = DataLoader()
    lines = loader.load_text_sequences()
    assert len(lines) >= 2
    assert all(isinstance(x, str) and x for x in lines)

