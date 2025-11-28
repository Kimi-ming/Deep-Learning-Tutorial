# -*- coding: utf-8 -*-
"""
Smoke Tests - 快速验证基本功能

这些测试应该快速运行（< 30秒），验证：
1. 所有模块可以成功导入
2. main.py 入口可以运行
3. 核心功能不崩溃
"""

import sys
import os
import glob
import importlib
import subprocess
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.smoke
class TestModuleImports:
    """测试所有教学模块可以成功导入"""

    def test_discover_modules(self):
        """动态发现所有教学模块"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)

        # 发现所有 deep_learning_*.py 文件
        module_files = glob.glob("deep_learning_*.py")
        assert len(module_files) > 0, "应该至少有一个教学模块"

        print(f"\n发现 {len(module_files)} 个教学模块")

    def test_import_all_modules(self):
        """测试所有模块可以成功导入"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)

        module_files = glob.glob("deep_learning_*.py")
        failed_imports = []

        for file_path in module_files:
            module_name = file_path[:-3]  # 移除 .py
            try:
                module = importlib.import_module(module_name)
                assert module is not None
                print(f"✓ {module_name}")
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                print(f"✗ {module_name}: {e}")

        assert len(failed_imports) == 0, f"导入失败的模块: {failed_imports}"

    def test_modules_have_docstrings(self):
        """测试主要教学模块都有文档字符串"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)

        # 仅检查非练习模块
        module_files = [f for f in glob.glob("deep_learning_*.py")
                       if 'exercises' not in f]
        modules_without_docs = []

        for file_path in module_files:
            module_name = file_path[:-3]
            try:
                module = importlib.import_module(module_name)
                if not module.__doc__ or not module.__doc__.strip():
                    modules_without_docs.append(module_name)
            except Exception:
                pass

        assert len(modules_without_docs) == 0, \
            f"缺少文档字符串的模块: {modules_without_docs}"


@pytest.mark.smoke
class TestMainEntry:
    """测试 main.py 入口点"""

    def test_main_py_exists(self):
        """测试 main.py 文件存在"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_path = os.path.join(root_dir, "main.py")
        assert os.path.exists(main_path), "main.py 应该存在"

    def test_main_help(self):
        """测试 main.py --help 不崩溃"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)

        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"main.py --help 应该成功运行\nstderr: {result.stderr}"
        # 检查中文"用法"或英文"usage"
        assert "用法" in result.stdout or "usage" in result.stdout.lower(), \
            "帮助信息应该包含使用说明"

    def test_main_list(self):
        """测试 main.py --list 不崩溃"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)

        result = subprocess.run(
            [sys.executable, "main.py", "--list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"main.py --list 应该成功运行\nstderr: {result.stderr}"
        assert "deep_learning" in result.stdout, \
            "列表应该包含模块名称"
        assert "模块加载失败" not in result.stdout, \
            "不应该有模块加载失败"

    def test_new_package_imports(self):
        """测试新包结构可以导入"""
        import deep_learning
        from deep_learning.fundamentals import Perceptron, MLP, DeepNetwork
        from deep_learning.architectures import SimpleCNN, SimpleRNN, transformer_architecture
        from deep_learning.optimizers import sgd_step, adam_update

        assert hasattr(deep_learning, 'fundamentals')
        assert callable(sgd_step)
        assert callable(adam_update)
        assert Perceptron is not None
        assert MLP is not None
        assert DeepNetwork is not None
        assert SimpleCNN is not None
        assert SimpleRNN is not None
        assert callable(transformer_architecture)


@pytest.mark.smoke
class TestCoreFunctionality:
    """测试核心功能不崩溃"""

    def test_deep_network(self):
        """测试深度网络可以创建"""
        from deep_learning_fundamentals import DeepNetwork

        # 创建网络
        network = DeepNetwork(layers=[2, 3, 1], learning_rate=0.1)

        assert network is not None
        assert network.layers == [2, 3, 1]

        print("✓ 深度网络创建正常")

    def test_simple_cnn(self):
        """测试简单CNN可以创建"""
        from deep_learning_cnn import SimpleCNN

        # 创建 CNN
        cnn = SimpleCNN(
            input_shape=(5, 5, 1),
            conv_filters=[(2, 3, 1)],
            fc_layers=[10]
        )

        assert cnn is not None
        assert hasattr(cnn, 'forward')

        print("✓ CNN 创建正常")

    def test_simple_rnn(self):
        """测试简单RNN可以创建"""
        from deep_learning_rnn import SimpleRNN

        # 创建 RNN
        rnn = SimpleRNN(
            input_size=2,
            hidden_size=3,
            output_size=1
        )

        assert rnn is not None
        assert rnn.input_size == 2
        assert rnn.hidden_size == 3

        print("✓ RNN 创建正常")


@pytest.mark.smoke
def test_overall_smoke():
    """整体冒烟测试 - 确保测试套件可以运行"""
    import time
    start_time = time.time()

    # 这个测试应该很快完成
    assert True, "基本断言应该通过"

    elapsed = time.time() - start_time
    assert elapsed < 1.0, "基本测试应该在1秒内完成"

    print(f"\n✓ 冒烟测试套件运行正常 (耗时: {elapsed:.3f}s)")


if __name__ == "__main__":
    # 可以直接运行此文件进行快速测试
    pytest.main([__file__, "-v", "-m", "smoke"])
