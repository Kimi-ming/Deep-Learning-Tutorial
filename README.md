# Deep Learning Tutorial

纯Python实现的深度学习教学项目，不依赖TensorFlow/PyTorch等框架，帮助理解深度学习核心原理。

## 项目特点

- **纯Python实现**: 所有核心算法使用纯Python编写，便于理解底层原理
- **详细中文注释**: 每个模块都包含详细的中文注释和数学公式说明
- **完整教学体系**: 从感知机到Transformer，覆盖深度学习主要架构
- **可运行示例**: 所有模块都包含可直接运行的示例代码

## 核心内容

### 基础模块
- **感知机 (Perceptron)**: 最基础的神经网络单元
- **多层感知机 (MLP)**: 全连接神经网络实现
- **激活函数**: Sigmoid, ReLU, Tanh, Softmax

### 深度架构
- **卷积神经网络 (CNN)**: 卷积层、池化层实现
- **循环神经网络 (RNN/LSTM/GRU)**: 序列处理架构
- **Transformer**: 注意力机制和自注意力实现

### 高级主题
- **优化算法**: SGD, Adam, RMSprop, 学习率调度
- **生成模型**: GAN (生成对抗网络), VAE (变分自编码器)
- **神经架构搜索 (NAS)**: 自动化网络架构设计

## 环境要求

- Python >= 3.7
- NumPy >= 1.19.0 (核心依赖)
- Matplotlib >= 3.3.0 (可选，用于可视化)

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 开发环境（包含测试和代码质量工具）
pip install -r requirements-dev.txt
```

### 运行示例

```python
# 示例：训练感知机实现AND门
from deep_learning_fundamentals import Perceptron

# 创建感知机
perceptron = Perceptron(input_size=2)

# 训练数据：AND门
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

# 训练
for epoch in range(100):
    for xi, yi in zip(X, y):
        perceptron.train(xi, yi)

# 测试
for xi, yi in zip(X, y):
    pred = perceptron.predict(xi)
    print(f"Input: {xi}, Predicted: {pred}, Actual: {yi}")
```

## 项目结构

```
Deep-Learning-Tutorial/
├── deep_learning_fundamentals.py    # 基础：感知机、MLP
├── deep_learning_cnn.py             # 卷积神经网络
├── deep_learning_rnn.py             # 循环神经网络
├── deep_learning_advanced.py        # 高级主题：GAN、Transformer
├── deep_learning_advanced_optimization.py  # 优化算法
├── deep_learning_advanced_projects.py      # 高级项目
├── deep_learning_cutting_edge.py    # 前沿技术
├── deep_learning_math_theory.py     # 数学理论
├── deep_learning_exercises.py       # 练习题
├── DEEP_LEARNING_GUIDE.md           # 详细教学指南
├── requirements.txt                 # 依赖列表
└── README.md                        # 项目说明
```

## 开发命令

```bash
# 格式化代码
make format

# 运行代码检查
make lint

# 运行测试
make test

# 清理缓存
make clean
```

## 学习路径

1. **入门**: 从 `deep_learning_fundamentals.py` 开始，理解感知机和MLP
2. **深入**: 学习 CNN (`deep_learning_cnn.py`) 和 RNN (`deep_learning_rnn.py`)
3. **进阶**: 探索 Transformer 和 GAN (`deep_learning_advanced.py`)
4. **优化**: 学习各种优化算法 (`deep_learning_advanced_optimization.py`)
5. **实践**: 完成 `deep_learning_exercises.py` 中的练习

详细教学内容请参考 [DEEP_LEARNING_GUIDE.md](DEEP_LEARNING_GUIDE.md)

## 贡献

欢迎提交问题和改进建议！

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

本项目旨在教学目的，帮助理解深度学习原理。生产环境请使用成熟框架如 PyTorch 或 TensorFlow。
