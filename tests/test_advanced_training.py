# -*- coding: utf-8 -*-
"""
高级训练特性示例测试（分布式/混合精度 stub）
"""

import subprocess
import sys
import os


def test_distributed_stub_runs():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "examples.06_distributed_stub"]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "预测" in result.stdout


def test_mixed_precision_stub_runs():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "examples.07_mixed_precision_stub"]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "混合精度模拟完成" in result.stdout
