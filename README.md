# Deep Learning Tutorial

纯 Python 的深度学习教学项目（无外部框架），涵盖从感知机到 Transformer 的核心原理与示例。

English version: see `i18n/README_en.md`.

## 特点
- 纯 Python 实现，便于理解底层计算
- 新包结构 `deep_learning/`，模块化组织
- 完整示例与数据加载小样本，配套测试
- 可视化与性能工具（EarlyStopping、梯度累积）
- 可选 NumPy 加速后端 + 模型保存/加载工具（pickle/JSON）
- 文档：Sphinx API 文档（`make docs`），Docker 支持、云部署指南

## 环境与安装
- Python >= 3.7
- NumPy >= 1.19.0（核心）
- Matplotlib >= 3.3.0（可选，可视化）

- `pip install .` 或 `pip install -r requirements.txt`（运行时）
- `pip install -r requirements-dev.txt`（开发/测试）
- 容器：`docker build -t dl-tutorial .` 或 `docker-compose run tutorial`

## 使用方式
### 1) CLI 菜单
```bash
python main.py          # 交互式选择包内模块
python main.py --list   # 列出包内模块
python main.py --help   # 查看帮助
# 安装后可直接使用
dl-tutorial             # 等价于 python -m main
docker-compose run tutorial  # 容器方式列出模块
```

### 2) 代码导入（推荐新包）
```python
from deep_learning.fundamentals import Perceptron

perceptron = Perceptron(input_size=2)
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
for _ in range(100):
    perceptron.train(X, y)
print([perceptron.predict(xi) for xi in X])
```

### 3) 运行示例
```bash
python -m examples.01_perceptron_and_gate
python -m examples.03_cnn_edge_detection
```

### 4) 运行测试
```bash
pytest -q          # 全部测试
make test          # 等价
```

## 目录结构（精简）
```
deep_learning/              # 主包（推荐）
├─ fundamentals/            # Perceptron, DeepNetwork, MLP
├─ architectures/           # CNN, RNN, Transformer 演示
├─ optimizers/              # SGD/Adam/调度器 + 高级优化
├─ advanced/                # GAN/Transformer 演示、项目示例
└─ utils/                   # 激活/损失/初始化/数学/性能/可视化
examples/                   # 01-05 示例脚本
datasets/                   # mnist_sample.npz, text_sequences.txt, data_loader.py
exercises/                  # 练习封装与占位答案
tests/                      # 覆盖包内模块/示例/数据/性能等
docs/ARCHITECTURE.md        # 包结构说明
README.md, TODO.md, CHANGELOG.md
```

> 说明：根目录的 `deep_learning_*.py` 文件仅为兼容入口，后续版本将移除；请改用 `deep_learning/` 包路径。

## 学习路径（建议）
1. 基础：`deep_learning.fundamentals`（感知机/MLP）
2. 架构：`deep_learning.architectures`（CNN/RNN/Transformer 演示）
3. 优化：`deep_learning.optimizers`（SGD/Adam/调度器）
4. 高级：`deep_learning.advanced`（GAN/项目演示）
5. 实践：`examples/`、`exercises/`、`datasets/` 小样本

更多细节参见 [DEEP_LEARNING_GUIDE.md](DEEP_LEARNING_GUIDE.md) 与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
API 文档见 Sphinx 构建输出（`make docs`）或在 docs/_build 下查看。

## 贡献
- 提交 Issue/PR 前请运行 `make format && make lint && make test`
- 代码与文档保持一致，优先使用新包导入

## 许可证
MIT，详见 [LICENSE](LICENSE)。
