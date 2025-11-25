# 常见问题 (FAQ)

本文档收集了使用 Deep Learning Tutorial 时的常见问题和解决方案。

## 安装和环境

### Q1: 需要什么 Python 版本？

**A:** 项目需要 Python >= 3.7

检查您的 Python 版本：
```bash
python --version
# 或
python3 --version
```

如果版本过低，请从 [python.org](https://www.python.org/downloads/) 下载最新版本。

### Q2: 如何安装依赖？

**A:** 使用以下命令安装：

```bash
# 基础依赖（仅 numpy）
pip install -r requirements.txt

# 开发依赖（包含测试和代码质量工具）
pip install -r requirements-dev.txt

# 或使用 make 命令
make install        # 生产依赖
make install-dev    # 开发依赖
```

### Q3: 安装 numpy 时出错怎么办？

**A:** 尝试以下解决方案：

1. **升级 pip**
   ```bash
   pip install --upgrade pip
   ```

2. **使用国内镜像**
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
   ```

3. **使用 conda**（推荐）
   ```bash
   conda install numpy
   ```

4. **在 macOS 上**
   如果遇到编译错误，可能需要安装 Xcode Command Line Tools：
   ```bash
   xcode-select --install
   ```

### Q4: 是否需要 GPU？

**A:** 不需要！本项目是纯 Python 实现，用于教学目的，在 CPU 上运行即可。

如果您需要训练大规模模型，建议使用 PyTorch 或 TensorFlow 等框架。

### Q5: matplotlib 安装失败怎么办？

**A:** matplotlib 是可选依赖，仅用于可视化。如果不需要可视化功能，可以跳过：

```bash
# 仅安装 numpy
pip install "numpy>=1.19.0,<2.0"
```

如果需要安装 matplotlib：
```bash
# 使用 conda（推荐）
conda install matplotlib

# 或使用镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
```

## 使用问题

### Q6: 如何运行教学模块？

**A:** 有三种方式：

1. **交互式菜单（推荐）**
   ```bash
   python main.py
   ```

2. **命令行模式**
   ```bash
   python main.py --list          # 列出所有模块
   python main.py fundamentals    # 运行指定模块
   ```

3. **直接运行**
   ```bash
   python deep_learning_fundamentals.py
   ```

### Q7: 运行模块时报错 "ModuleNotFoundError"

**A:** 确保：

1. 已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 在项目根目录下运行：
   ```bash
   cd /path/to/Deep-Learning-Tutorial
   python main.py
   ```

3. 如果使用虚拟环境，确保已激活：
   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

### Q8: 如何查看模块详细信息？

**A:** 在交互式菜单中输入 `help`：

```bash
python main.py
# 在菜单中输入: help
```

或者查看模块的文档字符串：
```python
import deep_learning_fundamentals
help(deep_learning_fundamentals)
```

### Q9: 训练速度很慢怎么办？

**A:** 本项目使用纯 Python 实现，性能有限，这是正常的。建议：

1. **减少训练数据量**
   - 用于理解算法原理，不需要大规模数据集

2. **降低迭代次数**
   - 观察学习过程即可，不需要完全收敛

3. **使用更快的实现**
   - 对于实际应用，请使用 PyTorch 或 TensorFlow

### Q10: 如何保存训练好的模型？

**A:** 当前版本暂不支持模型保存。可以自己实现：

```python
import pickle

# 保存模型
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 代码相关

### Q11: 为什么不使用 NumPy 进行矩阵运算？

**A:** 本项目的目标是教学，帮助理解算法底层实现。使用纯 Python 列表可以：

- 清晰展示算法逻辑
- 避免隐藏实现细节
- 降低学习门槛

在 Phase 6 (远期规划) 中可能会添加 NumPy 加速版本作为对比。

### Q12: 代码中的数学公式在哪里？

**A:** 数学公式在代码注释中：

```python
# 前向传播: y = Wx + b
# 其中 W 是权重矩阵，x 是输入，b 是偏置
```

也可以查看：
- `docs/ARCHITECTURE.md` - 架构设计
- `DEEP_LEARNING_GUIDE.md` - 学习指南
- 代码中的 docstring

### Q13: 如何调试代码？

**A:** 推荐方式：

1. **使用 print 调试**
   ```python
   print(f"权重: {weights}")
   print(f"损失: {loss}")
   ```

2. **使用 Python 调试器**
   ```python
   import pdb
   pdb.set_trace()  # 设置断点
   ```

3. **使用 IDE 调试器**
   - VSCode: 设置断点，按 F5
   - PyCharm: 设置断点，点击 Debug

### Q14: 发现代码 bug 怎么办？

**A:** 欢迎报告 bug！

1. 检查是否已有相关 [Issue](https://github.com/Kimi-ming/Deep-Learning-Tutorial/issues)
2. 创建新 Issue，包含：
   - 问题描述
   - 重现步骤
   - 期望结果vs实际结果
   - Python 版本和操作系统

## 学习相关

### Q15: 建议的学习顺序是什么？

**A:** 推荐顺序：

1. **基础**：`deep_learning_fundamentals.py` - 理解感知机和 MLP
2. **CNN**：`deep_learning_cnn.py` - 学习卷积神经网络
3. **RNN**：`deep_learning_rnn.py` - 理解循环神经网络
4. **优化**：`deep_learning_advanced_optimization.py` - 学习优化算法
5. **高级**：`deep_learning_advanced.py` - 探索 Transformer, GAN 等
6. **练习**：`deep_learning_exercises.py` - 动手实践

### Q16: 需要什么数学基础？

**A:** 建议掌握：

- **线性代数**：矩阵乘法、向量运算
- **微积分**：导数、链式法则、梯度
- **概率统计**：基础概率、期望、方差

不需要高深的数学知识，代码中会解释关键概念。

### Q17: 学完后可以做什么？

**A:** 学完本教程后，您将：

- 理解深度学习的底层原理
- 能够从零实现基础神经网络
- 为学习 PyTorch/TensorFlow 打下坚实基础
- 能够阅读和理解深度学习论文

下一步建议：
- 学习 PyTorch 或 TensorFlow
- 在 Kaggle 上做实际项目
- 阅读经典论文（如 ResNet, Transformer）

### Q18: 与在线课程相比，本项目的优势是什么？

**A:** 本项目的特点：

- ✅ 纯 Python 实现，代码简洁易懂
- ✅ 中文注释，详细的文档说明
- ✅ 可以直接运行和修改代码
- ✅ 专注于算法原理，而非框架使用
- ✅ 完全免费开源

适合作为在线课程的补充学习材料。

## 贡献相关

### Q19: 如何为项目做贡献？

**A:** 请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

简要步骤：
1. Fork 项目
2. 创建分支
3. 进行修改
4. 运行测试和代码检查
5. 提交 Pull Request

### Q20: 代码规范是什么？

**A:** 我们使用：

- **Black** 进行代码格式化 (line-length=100)
- **Ruff** 进行代码检查 (E, F, I, W 规则)
- **Pytest** 进行测试

运行命令：
```bash
make format  # 格式化代码
make lint    # 检查代码
make test    # 运行测试
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 其他问题

### Q21: 项目的未来计划是什么？

**A:** 详见 [TODO.md](TODO.md)，主要包括：

- Phase 2: 包结构重构
- Phase 3: 添加测试框架
- Phase 4-5: 代码重构和模块化
- Phase 6: 添加更多示例和可视化
- Phase 7: 性能优化

### Q22: 可以用于商业项目吗？

**A:** 可以！本项目使用 MIT 许可证，允许：

- ✅ 商业使用
- ✅ 修改
- ✅ 分发
- ✅ 私有使用

但需要：
- 保留版权声明
- 保留许可证文本

详见 [LICENSE](LICENSE)

### Q23: 如何获取更新？

**A:**

1. **Watch 项目**：在 GitHub 上点击 "Watch" 按钮
2. **Star 项目**：关注项目更新
3. **定期拉取**：
   ```bash
   git pull origin main
   ```

### Q24: 找不到问题的答案怎么办？

**A:**

1. 搜索现有 [Issues](https://github.com/Kimi-ming/Deep-Learning-Tutorial/issues)
2. 查看 [DEEP_LEARNING_GUIDE.md](DEEP_LEARNING_GUIDE.md)
3. 查看 [CONTRIBUTING.md](CONTRIBUTING.md)
4. 创建新 Issue 提问

---

如果您的问题没有在这里找到答案，欢迎创建 Issue！我们会及时更新此文档。
