# -*- coding: utf-8 -*-
"""
GAN 简化实现测试
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from deep_learning.advanced import SimpleGAN


def test_simple_gan_shapes_and_train():
    gan = SimpleGAN(noise_dim=2, data_dim=2, lr=0.1)
    noise = [0.5, -0.5]
    real = [1.0, 0.5]

    fake = gan.generator(noise)
    assert len(fake) == 2

    real_score, fake_score = gan.train_step(real, noise)
    assert isinstance(real_score, float)
    assert isinstance(fake_score, float)
