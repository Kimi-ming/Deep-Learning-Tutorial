# 项目架构设计

本文档定义了 Deep Learning Tutorial 项目的包结构、模块划分和迁移计划。

## 目录结构

```
Deep-Learning-Tutorial/
├── deep_learning/              # 主包目录
│   ├── __init__.py            # 包初始化，导出公共API
│   ├── utils/                 # 公共工具模块
│   │   ├── __init__.py
│   │   ├── activations.py     # 激活函数 (sigmoid, relu, tanh, softmax)
│   │   ├── losses.py          # 损失函数 (mse, cross_entropy, binary_cross_entropy)
│   │   ├── initializers.py    # 权重初始化 (xavier, he, uniform)
│   │   ├── math_ops.py        # 数学操作 (矩阵乘法, 梯度裁剪, 归一化)
│   │   └── visualization.py   # 可视化工具 (损失曲线, 准确率曲线)
│   ├── fundamentals/          # 基础模块
│   │   ├── __init__.py
│   │   ├── perceptron.py      # 感知机
│   │   └── mlp.py             # 多层感知机
│   ├── architectures/         # 深度学习架构
│   │   ├── __init__.py
│   │   ├── cnn.py             # 卷积神经网络
│   │   ├── rnn.py             # 循环神经网络 (RNN, LSTM, GRU)
│   │   └── transformer.py     # Transformer架构
│   ├── optimizers/            # 优化器
│   │   ├── __init__.py
│   │   ├── sgd.py             # 随机梯度下降
│   │   ├── adam.py            # Adam优化器
│   │   └── schedulers.py      # 学习率调度器
│   └── advanced/              # 高级主题
│       ├── __init__.py
│       ├── gan.py             # 生成对抗网络
│       ├── vae.py             # 变分自编码器
│       └── nas.py             # 神经架构搜索
├── tests/                     # 测试目录
│   ├── __init__.py
│   ├── test_smoke.py          # 冒烟测试
│   ├── test_fundamentals.py  # 基础模块测试
│   ├── test_cnn.py            # CNN测试
│   ├── test_rnn.py            # RNN测试
│   └── utils/                 # 工具测试
│       ├── test_activations.py
│       ├── test_losses.py
│       └── test_visualization.py
├── examples/                  # 独立示例
│   ├── 01_perceptron_and_gate.py
│   ├── 02_mlp_xor_gate.py
│   ├── 03_cnn_edge_detection.py
│   ├── 04_rnn_sequence_memory.py
│   └── 05_transformer_attention.py
├── exercises/                 # 练习题目录
│   ├── exercise_01.py
│   ├── exercise_02.py
│   └── solutions/             # 练习答案
├── datasets/                  # 示例数据集
│   ├── mnist_sample.npz
│   └── text_sequences.txt
├── docs/                      # 文档目录
│   ├── ARCHITECTURE.md        # 本文档
│   └── API.md                 # API文档
├── main.py                    # CLI入口程序
├── deep_learning_fundamentals.py    # [已废弃] 迁移到 deep_learning/fundamentals/
├── deep_learning_cnn.py              # [已废弃] 迁移到 deep_learning/architectures/
├── deep_learning_rnn.py              # [已废弃] 迁移到 deep_learning/architectures/
├── deep_learning_advanced.py         # [已废弃] 迁移到 deep_learning/advanced/
└── ...                        # 其他已废弃文件
```

## 模块划分

### 1. utils/ - 公共工具模块

**职责**: 提供可重用的底层工具函数

**模块说明**:
- `activations.py`: 所有激活函数的实现
  - `sigmoid(x)`: Sigmoid激活函数，数值稳定版本
  - `relu(x)`: ReLU激活函数
  - `tanh(x)`: Tanh激活函数
  - `softmax(x)`: Softmax激活函数，数值稳定版本

- `losses.py`: 损失函数实现
  - `mse(y_true, y_pred)`: 均方误差
  - `cross_entropy(y_true, y_pred, epsilon=1e-15)`: 交叉熵
  - `binary_cross_entropy(y_true, y_pred, epsilon=1e-15)`: 二元交叉熵

- `initializers.py`: 权重初始化策略
  - `xavier_init(shape)`: Xavier/Glorot初始化
  - `he_init(shape)`: He初始化
  - `uniform_init(shape, scale=0.01)`: 均匀分布初始化

- `math_ops.py`: 数学操作工具
  - `matmul(a, b)`: 矩阵乘法
  - `clip_gradients(grads, max_norm)`: 梯度裁剪
  - `normalize(x, axis)`: 归一化

- `visualization.py`: 可视化工具
  - `plot_loss_curve(losses, title)`: 绘制损失曲线
  - `plot_accuracy_curve(accuracies, title)`: 绘制准确率曲线

### 2. fundamentals/ - 基础模块

**职责**: 提供神经网络的基础组件

**模块说明**:
- `perceptron.py`: 单层感知机实现
  - `Perceptron` 类: 基础感知机，用于线性可分问题

- `mlp.py`: 多层感知机实现
  - `MLP` 类: 全连接神经网络

### 3. architectures/ - 深度学习架构

**职责**: 实现经典深度学习架构

**模块说明**:
- `cnn.py`: 卷积神经网络
  - `Conv2D` 类: 2D卷积层
  - `MaxPooling2D` 类: 最大池化层
  - `CNN` 类: 完整CNN网络

- `rnn.py`: 循环神经网络
  - `RNN` 类: 基础RNN
  - `LSTM` 类: 长短期记忆网络
  - `GRU` 类: 门控循环单元

- `transformer.py`: Transformer架构
  - `MultiHeadAttention` 类: 多头注意力
  - `Transformer` 类: 完整Transformer

### 4. optimizers/ - 优化器

**职责**: 提供各种优化算法

**模块说明**:
- `sgd.py`: 随机梯度下降
  - `SGD` 类: 基础SGD优化器
  - `SGDMomentum` 类: 带动量的SGD

- `adam.py`: Adam优化器
  - `Adam` 类: Adam优化器
  - `RMSprop` 类: RMSprop优化器

- `schedulers.py`: 学习率调度
  - `StepLR` 类: 阶梯式学习率
  - `ExponentialLR` 类: 指数衰减学习率
  - `CosineAnnealingLR` 类: 余弦退火学习率

### 5. advanced/ - 高级主题

**职责**: 实现高级深度学习技术

**模块说明**:
- `gan.py`: 生成对抗网络
  - `Generator` 类: 生成器
  - `Discriminator` 类: 判别器
  - `GAN` 类: 完整GAN

- `vae.py`: 变分自编码器
  - `VAE` 类: 变分自编码器实现

- `nas.py`: 神经架构搜索
  - `NAS` 类: 基础NAS实现

## 导入路径约定

### 公共API导入

用户应该从顶层包导入常用类和函数：

```python
# 推荐的导入方式
from deep_learning import Perceptron, MLP
from deep_learning import CNN, RNN, LSTM
from deep_learning import SGD, Adam
from deep_learning.utils import sigmoid, relu, mse, cross_entropy

# 避免的导入方式（内部实现可能变更）
from deep_learning.fundamentals.perceptron import Perceptron  # 不推荐
```

### 顶层 __init__.py 导出

`deep_learning/__init__.py` 应导出常用类和函数：

```python
# deep_learning/__init__.py

from .fundamentals.perceptron import Perceptron
from .fundamentals.mlp import MLP
from .architectures.cnn import CNN, Conv2D, MaxPooling2D
from .architectures.rnn import RNN, LSTM, GRU
from .architectures.transformer import Transformer
from .optimizers.sgd import SGD
from .optimizers.adam import Adam
from .advanced.gan import GAN
from .advanced.vae import VAE

__all__ = [
    'Perceptron', 'MLP',
    'CNN', 'Conv2D', 'MaxPooling2D',
    'RNN', 'LSTM', 'GRU',
    'Transformer',
    'SGD', 'Adam',
    'GAN', 'VAE',
]
```

## 迁移计划

### Phase 4: 代码重构

1. 创建 `deep_learning/utils/` 目录
2. 从现有文件中提取公共工具函数
3. 添加数值稳定性改进
4. 编写单元测试验证工具函数

### Phase 5: 模块迁移

按以下顺序迁移现有代码：

1. **deep_learning_fundamentals.py** → `deep_learning/fundamentals/`
   - 提取 `Perceptron` → `perceptron.py`
   - 提取 `MLP` → `mlp.py`
   - 更新导入使用 `utils` 模块

2. **deep_learning_cnn.py** → `deep_learning/architectures/cnn.py`
   - 整合卷积、池化层
   - 使用 `utils` 中的激活函数

3. **deep_learning_rnn.py** → `deep_learning/architectures/rnn.py`
   - 整合 RNN, LSTM, GRU
   - 使用 `utils.math_ops.clip_gradients`

4. **deep_learning_advanced_optimization.py** → `deep_learning/optimizers/`
   - 提取 Adam, RMSprop → `adam.py`
   - 提取学习率调度 → `schedulers.py`

5. **deep_learning_advanced.py** → `deep_learning/advanced/`
   - 提取 GAN → `gan.py`
   - 提取 Transformer → `architectures/transformer.py`

6. **deep_learning_advanced_projects.py** → `deep_learning/advanced/`
   - 提取 VAE → `vae.py`
   - 提取 NAS → `nas.py`

### 向后兼容

在过渡期（1-2个版本）：
- 保留旧文件，添加 `DEPRECATED` 警告
- 旧文件内部调用新包代码
- 在文档中标注新的导入路径

示例：
```python
# deep_learning_fundamentals.py (已废弃)
import warnings
warnings.warn(
    "deep_learning_fundamentals.py is deprecated. "
    "Use 'from deep_learning import Perceptron, MLP' instead.",
    DeprecationWarning
)

# 重新导出新包中的类
from deep_learning.fundamentals.perceptron import Perceptron
from deep_learning.fundamentals.mlp import MLP
```

## 依赖关系

### 模块依赖图

```
utils/ (无依赖)
  ↓
fundamentals/ (依赖 utils)
  ↓
architectures/ (依赖 utils, fundamentals)
  ↓
optimizers/ (依赖 utils)
  ↓
advanced/ (依赖 utils, fundamentals, architectures, optimizers)
```

### 设计原则

1. **单向依赖**: 低层模块不依赖高层模块
2. **最小依赖**: 只导入必需的模块
3. **循环依赖避免**: 通过接口和依赖注入避免循环依赖
4. **测试隔离**: 每个模块可独立测试

## API 稳定性

### 稳定 API（保证向后兼容）

- `deep_learning/__init__.py` 导出的所有类和函数
- `deep_learning.utils` 下的所有公共函数
- 类的公共方法（非 `_` 开头）

### 不稳定 API（可能变更）

- 内部实现细节（`_` 开头的方法和属性）
- 直接从子模块导入的类（如 `from deep_learning.fundamentals.perceptron import ...`）
- 未在文档中明确说明的功能

## 测试策略

### 单元测试

每个模块都有对应的测试文件：
- `tests/test_fundamentals.py` 测试 `deep_learning/fundamentals/`
- `tests/test_cnn.py` 测试 `deep_learning/architectures/cnn.py`
- 等等

### 集成测试

- `tests/test_smoke.py`: 验证所有模块可成功导入
- `examples/` 目录下的示例作为端到端测试

### 测试覆盖率目标

- 核心工具函数 (utils): 90%+
- 基础模块 (fundamentals): 80%+
- 架构模块 (architectures): 70%+

## 版本计划

- **v0.1.0**: 当前基线版本（单文件结构）
- **v0.2.0**: 完成 utils 提取和基础模块迁移
- **v0.3.0**: 完成所有模块迁移
- **v0.4.0**: 删除已废弃文件，完全采用新结构
- **v1.0.0**: 稳定版本发布

## 参考资源

- [Python 包结构最佳实践](https://docs.python-guide.org/writing/structure/)
- [语义化版本](https://semver.org/lang/zh-CN/)
- [API 设计指南](https://www.python.org/dev/peps/pep-0008/)
