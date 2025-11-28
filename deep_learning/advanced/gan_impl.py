# -*- coding: utf-8 -*-
"""
GAN 简化实现（教学用）
"""

import random


class SimpleGAN:
    """
    简化版 GAN，用于演示训练流程，不包含完整反向传播。
    """

    def __init__(self, noise_dim=2, data_dim=2, lr=0.01):
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.lr = lr
        # 参数占位（使用简单向量表示）
        self.generator_weights = [random.uniform(-0.1, 0.1) for _ in range(noise_dim * data_dim)]
        self.discriminator_weights = [random.uniform(-0.1, 0.1) for _ in range(data_dim)]

    def generator(self, noise):
        """线性映射噪声 -> 数据空间"""
        if len(noise) != self.noise_dim:
            raise ValueError("noise 维度不匹配")
        # 简单线性组合
        output = []
        for i in range(self.data_dim):
            s = 0.0
            for j in range(self.noise_dim):
                s += noise[j] * self.generator_weights[i * self.noise_dim + j]
            output.append(s)
        return output

    def discriminator(self, sample):
        """线性评分"""
        if len(sample) != self.data_dim:
            raise ValueError("sample 维度不匹配")
        score = sum(w * x for w, x in zip(self.discriminator_weights, sample))
        return score

    def train_step(self, real_sample, noise):
        """
        演示性训练步（不做真实反传），只返回示例值。
        """
        fake = self.generator(noise)
        real_score = self.discriminator(real_sample)
        fake_score = self.discriminator(fake)
        # 简单梯度近似（符号取反）
        self.discriminator_weights = [w + self.lr * (r - f) for w, r, f in zip(self.discriminator_weights, real_sample, fake)]
        self.generator_weights = [w - self.lr * fs for w, fs in zip(self.generator_weights, [fake_score] * len(self.generator_weights))]
        return real_score, fake_score


__all__ = ["SimpleGAN"]
