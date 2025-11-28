# -*- coding: utf-8 -*-
"""
数据加载与预处理工具（轻量版）
"""

import os
import numpy as np


class DataLoader:
    """简单的数据加载器"""

    def __init__(self, root: str = "datasets"):
        self.root = root

    def load_mnist_sample(self):
        """
        加载示例 MNIST 小样本 (mnist_sample.npz)

        Returns:
            images: (N, 28, 28) uint8
            labels: (N,) uint8
        """
        path = os.path.join(self.root, "mnist_sample.npz")
        if not os.path.exists(path):
            raise FileNotFoundError("未找到 mnist_sample.npz，请确认 datasets/ 目录存在示例文件")
        data = np.load(path)
        return data["images"], data["labels"]

    def load_text_sequences(self):
        """
        加载简单文本序列示例
        """
        path = os.path.join(self.root, "text_sequences.txt")
        if not os.path.exists(path):
            raise FileNotFoundError("未找到 text_sequences.txt，请确认 datasets/ 目录存在示例文件")
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines


__all__ = ["DataLoader"]
