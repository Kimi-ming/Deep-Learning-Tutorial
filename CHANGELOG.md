# Changelog

所有重要的项目变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中
- 模块化重构（Phase 2-5）
- 测试框架（Phase 3）
- 独立示例（Phase 6）
- 性能优化（Phase 7）

## [0.1.0] - 2024-11-25

### 新增
- 项目基础文件
  - .gitignore (Python, 虚拟环境, IDE, 深度学习文件)
  - LICENSE (MIT)
  - requirements.txt (numpy依赖)
  - requirements-dev.txt (测试和代码质量工具)
  - pyproject.toml (black, ruff, pytest配置)
  - Makefile (format, lint, test命令)
  - README.md (项目说明和快速开始)
  - CHANGELOG.md (变更日志)

### 已实现的核心模块
- `deep_learning_fundamentals.py`: 感知机、MLP、激活函数、损失函数
- `deep_learning_cnn.py`: CNN、卷积层、池化层
- `deep_learning_rnn.py`: RNN、LSTM、GRU
- `deep_learning_advanced.py`: GAN、Transformer、强化学习、元学习
- `deep_learning_advanced_optimization.py`: Adam、RMSprop、学习率调度
- `deep_learning_advanced_projects.py`: NAS、VAE、MAML
- `deep_learning_cutting_edge.py`: ViT、EfficientNet、MoE
- `deep_learning_math_theory.py`: 链式法则、信息论、优化理论
- `deep_learning_exercises.py`: 6个实践练习
- `DEEP_LEARNING_GUIDE.md`: 详细教学指南

### 说明
这是项目的基线版本，包含所有核心教学模块的初始实现。
后续版本将进行模块化重构、添加测试、优化代码质量。

---

## 版本说明

- **[Unreleased]**: 未发布的变更
- **[0.1.0]**: 基线版本，包含所有核心功能的初始实现

## 链接

- [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
- [语义化版本](https://semver.org/lang/zh-CN/)
