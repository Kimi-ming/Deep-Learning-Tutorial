# Deep Learning Tutorial - TODO List

## 🔴 Phase 1: 基础与规范 (立即执行，无依赖)

### 基础文件
- [x] 创建 .gitignore (虚拟环境、缓存、检查点、数据集)
- [x] 创建 LICENSE (MIT)
- [x] 创建 requirements.txt (numpy>=1.19.0,<2.0; 可选: matplotlib>=3.3.0)
- [x] 创建 requirements-dev.txt (pytest>=7.0.0, black>=22.0.0, ruff>=0.1.0)

### 代码规范
- [x] 选择并记录格式化工具 (推荐 black, line-length=100)
- [x] 选择并记录静态检查工具 (推荐 ruff, 规则: E,F,I)
- [x] 创建 pyproject.toml (配置 black 和 ruff)
- [x] 创建 Makefile (make format, make lint, make test)

### 核心文档
- [x] 创建 README.md
  - 项目目标与特点
  - 环境要求 (Python >= 3.7)
  - 快速开始 (安装、运行示例)
  - 当前文件结构说明
- [x] 创建 CHANGELOG.md (记录当前基线版本)

---

## ✅ Phase 2: 包结构与入口 (依赖 Phase 1) - 已完成

### 设计文档
- [x] 创建 docs/ARCHITECTURE.md
  - 定义 `deep_learning/` 包结构
  - 子模块划分: utils/fundamentals/architectures/optimizers/advanced
  - 导入路径约定 (避免后续重构破坏)
  - 迁移计划与依赖关系

### 最小入口
- [x] 创建 main.py (CLI/交互式菜单)
  - 动态发现并列出当前所有可执行的教学模块
  - 支持数字选择运行
  - 最小可运行示例

### 配套文档
- [x] 更新 DEEP_LEARNING_GUIDE.md (与包结构/入口保持一致)
- [x] 创建 CONTRIBUTING.md (环境搭建、格式化、测试命令)
- [x] 创建 FAQ.md (依赖安装、环境问题、常见错误)

### Bug 修复
- [x] 修复 main.py 模块加载 bug (PR #3)
  - 安全处理 __doc__ 为 None 的情况
  - 优化异常信息输出
  - 为所有教学模块添加 docstring

---

## ✅ Phase 3: 测试骨架 (依赖 Phase 2，确保包结构已定) - 已完成

### 测试框架搭建
- [x] 创建 tests/ 目录和 tests/__init__.py
- [x] 配置 pytest.ini (testpaths=tests, python_files=test_*.py)
- [x] 创建 tests/test_smoke.py (最小 smoke tests)
  - 动态测试当前所有可导入模块能成功导入 (适配当前.py文件或未来包结构)
  - 测试 main.py 入口能运行不崩溃
  - 测试至少3个核心功能 forward/训练一小步不崩溃
  - 运行时间 < 30秒

### 最小单元测试 (针对当前结构)
- [x] tests/test_fundamentals.py (DeepNetwork初始化、权重、激活函数)
- [x] tests/test_cnn.py (卷积输出形状、池化值验证)
- [x] tests/test_rnn.py (隐藏状态形状、权重初始化、激活函数)

### 预留测试任务 (Phase 4-6 新增功能的测试，仅标记待实现，不创建文件)
- [x] tests/utils/test_activations.py (Phase 4 已实现: 32个测试覆盖边界、数值稳定性)
- [x] tests/utils/test_losses.py (Phase 4 已实现: 18个测试覆盖epsilon处理、梯度验证)
- [x] tests/utils/test_visualization.py (Phase 6 已实现: matplotlib可用/不可用场景)
- [x] tests/test_data_loader.py (Phase 6 已实现: MNIST/文本加载、形状验证)
- [x] tests/test_examples.py (Phase 6 已实现: examples/ 下示例能运行不报错)

---

## ✅ Phase 4: 代码重构 (依赖 Phase 3，测试保护下重构) - 已完成

### 公共工具提取
- [x] 创建 deep_learning/utils/__init__.py
- [x] deep_learning/utils/activations.py
  - sigmoid (数值稳定, 范围裁剪 [-500, 500])
  - relu, tanh, softmax (数值稳定处理)
  - 每个函数包含 docstring + 数学公式
  - leaky_relu 和激活函数导数
- [x] deep_learning/utils/losses.py
  - mse, mae, cross_entropy, binary_cross_entropy
  - epsilon 参数防止 log(0)
  - sparse_categorical_cross_entropy, hinge, huber
- [x] deep_learning/utils/initializers.py (xavier, he, lecun, uniform, normal)
- [x] deep_learning/utils/math_ops.py (矩阵乘法、梯度裁剪、归一化、距离度量)

### 代码迁移与重构
- [x] 重构 deep_learning_fundamentals.py 使用 utils
  - 使用 he_normal 替换手动 He 初始化
  - 使用 relu, leaky_relu 替换内部实现
- [x] 重构 deep_learning_cnn.py 使用 utils
  - 使用 normal 替换 random.gauss 初始化
  - 使用 relu 替换内部方法
- [x] 重构 deep_learning_rnn.py 使用 utils
  - 使用 normal 替换权重初始化
  - 使用 tanh, softmax, matrix_vector_multiply
  - 导入 clip_gradients (注: 已导入但未应用到训练循环,留待 Phase 5 迁移时完善)
- [x] 减少重复代码约 45 行
- [x] 保持向后兼容 API

### 更新测试 (补充 Phase 3 预留的测试)
- [x] 实现 tests/utils/test_activations.py (边界测试、数值稳定性)
- [x] 实现 tests/utils/test_losses.py (epsilon处理、梯度验证)
- [x] 验证所有测试通过 (83个测试全部通过)
- [x] 确保重构未破坏任何现有功能

---

## ✅ Phase 5: 模块迁移 (依赖 Phase 4，新结构下迁移) - 已完成

### 创建新包结构
- [x] 创建 deep_learning/fundamentals/ (perceptron.py, mlp.py)
- [x] 创建 deep_learning/architectures/ (cnn.py, rnn.py, transformer.py)
- [x] 创建 deep_learning/optimizers/ (sgd.py, adam.py, schedulers.py)
- [x] 创建 deep_learning/advanced/ (gan.py, vae.py, nas.py)

### 迁移现有代码
- [x] 迁移 deep_learning_fundamentals.py → fundamentals/
  - Perceptron 类 → perceptron.py
  - MLP 类 → mlp.py
  - 使用 utils 公共代码
- [x] 迁移 deep_learning_cnn.py → architectures/cnn.py (使用公共 utils)
- [x] 迁移 deep_learning_rnn.py → architectures/rnn.py
  - 使用 utils.math_ops.clip_gradients 替换内部实现
  - 添加序列填充功能
  - 完善 LSTM 实现
- [x] 迁移 deep_learning_advanced_optimization.py → optimizers/ (提取 Adam, RMSprop 等)
- [x] 迁移 deep_learning_advanced.py → advanced/ (提取 GAN, Transformer)
- [x] 迁移 deep_learning_advanced_projects.py → advanced/ (提取 VAE, NAS)

### 兼容性处理
- [x] 旧文件添加 DEPRECATED 警告和注释
- [x] 旧文件调用新包代码 (向后兼容过渡期)

### 更新测试和文档
- [x] 更新所有测试导入路径
- [x] 验证所有测试通过
- [x] 更新 README.md 和 ARCHITECTURE.md

---

## 🔵 Phase 6: 示例与可视化 (依赖 Phase 5，新结构稳定后)

### 独立示例
- [x] 创建 examples/ 目录
- [x] examples/01_perceptron_and_gate.py (5-20行，完整流程)
- [x] examples/02_mlp_xor_gate.py (展示训练过程损失)
- [x] examples/03_cnn_edge_detection.py (边缘检测，ASCII可视化)
- [x] examples/04_rnn_sequence_memory.py (RNN记忆序列)
- [x] examples/05_transformer_attention.py (Transformer注意力机制)

### 练习重构
- [x] 重构 deep_learning_exercises.py → exercises/ 目录
- [x] 添加练习答案参考
- [x] 确保导入路径与新结构一致

### 基础可视化
- [x] deep_learning/utils/visualization.py
  - plot_loss_curve (损失曲线)
  - plot_accuracy_curve (准确率曲线)
  - try-except 处理 matplotlib 未安装
- [x] 在至少2个示例中使用可视化
- [x] 实现 tests/utils/test_visualization.py (matplotlib可用/不可用场景测试)

### 数据集支持
- [x] 创建 datasets/ 目录
- [x] 添加 MNIST 示例数据加载器 (100样本 .npz)
- [x] 添加文本序列示例数据
- [x] 创建 DataLoader 类 (加载、预处理)
- [x] 实现 tests/test_data_loader.py (MNIST/文本加载、形状验证)

### 示例验证测试
- [x] 实现 tests/test_examples.py
  - 测试 examples/ 下所有示例能运行不报错
  - 验证示例输出符合预期 (如 AND 门训练收敛)

---

## 🟣 Phase 7: 增强功能 (依赖 Phase 6，核心功能完成后)

### 特定模块增强
- [x] architectures/cnn.py: 添加更多卷积核示例 (Sobel, Laplacian等)
- [x] advanced/gan.py: 完善判别器和生成器训练逻辑（简化版 SimpleGAN）
- [x] advanced/transformer.py: 添加 Encoder-Decoder 完整实现（演示版）

### 性能优化
- [x] 全局添加批处理训练支持 (batch_size 参数, 适用于 MLP/CNN/RNN)
- [x] 优化关键路径循环 (矩阵乘法、卷积操作)
- [x] 添加早停机制 (EarlyStopping 类)
- [x] 实现梯度累积 (支持大batch模拟)

### 补充测试
- [x] tests/advanced/test_gan.py (生成器/判别器输出形状)
- [x] tests/test_performance.py (早停、梯度累积工具)

---

## ⚪ 远期/待讨论 (需确认是否实施)

### M1: 交互式功能
- [x] **决策**: 需要进度追踪系统 → 已创建 LearningProgress 类 + .progress.json 持久化 + 进度百分比
- [x] **决策**: 需要测验系统 → 已创建 Quiz 类 + 题库 JSON 支持 + 自动评分/结果报告
- [x] **决策**: 需要提示系统 → 已创建 HintSystem 类 + 分级提示 (默认3级)

### M2: Notebook 支持
- [x] **决策**: 转换为 Jupyter Notebooks → 已创建 notebooks/ + 核心教学模块示例 .ipynb + 简单交互演示

### M3: 国际化
- [x] **决策**: 需要英文版文档 → 已创建 i18n/README_en.md + 补充关键 docstring 英文描述

### M4: 打包发布
- [x] **决策**: 发布到 PyPI（可选，当前先完成打包脚手架）→ 已创建 setup.py + MANIFEST.in + 配置 dl-tutorial 入口（支持 `pip install .` 安装/CLI 运行）

### M5: CI/CD
- [x] **决策**: 需要 GitHub Actions → 已添加 .github/workflows/tests.yml (Python 3.8 & 3.11) 运行 pytest

### M6: 性能增强
- [ ] **决策**: 是否需要 NumPy 加速版本?
  - 如是 → deep_learning/utils/_numpy_backend.py + 性能对比文档 (保持纯Python默认)
  - 如否 → 仅保留纯Python实现
- [ ] **决策**: 是否需要模型保存/加载?
  - 如是 → 添加 save_model/load_model 函数 + JSON/pickle 序列化 + 示例
  - 如否 → 跳过

### M7: Docker/云平台
- [ ] **决策**: 是否需要 Docker 支持?
  - 如是 → 创建 Dockerfile + docker-compose.yml + 镜像构建文档
  - 如否 → 跳过
- [ ] **决策**: 是否需要云平台部署指南?
  - 如是 → 编写 AWS/GCP/Azure 部署文档
  - 如否 → 跳过

### M8: 文档增强
- [ ] **决策**: 是否生成 API 文档?
  - 如是 → 配置 Sphinx + 生成文档 + 托管到 Read the Docs
  - 如否 → 仅保留 docstring
- [ ] **决策**: 是否需要在线文档站点?
  - 如是 → 配置 GitHub Pages + mkdocs/sphinx
  - 如否 → 仅保留 Markdown 文档

### M9: 高级训练特性
- [ ] **决策**: 是否需要分布式训练示例?
  - 如是 → 添加多进程训练示例 (纯Python实现受限)
  - 如否 → 跳过
- [ ] **决策**: 是否需要混合精度训练?
  - 如是 → 添加 FP16 模拟示例 + 说明文档
  - 如否 → 跳过

---

## 📊 进度统计

| Phase | 主要任务 (checkbox) | 子任务/细项 (bullet) | 预估时间 | 状态 |
|-------|---------------------|---------------------|---------|------|
| Phase 1: 基础与规范 | 10项 | 6项 | 4-6h | ✅ 已完成 |
| Phase 2: 包结构与入口 | 6项 | 9项 | 4-6h | ✅ 已完成 |
| Phase 3: 测试骨架 | 9项 | 12项 | 6-8h | ✅ 已完成 |
| Phase 4: 代码重构 | 13项 | 11项 | 1-2天 | ✅ 已完成 |
| Phase 5: 模块迁移 | 12项 | 11项 | 2-3天 | ✅ 已完成 |
| Phase 6: 示例与可视化 | 17项 | 10项 | 1-2天 | ✅ 已完成 |
| Phase 7: 增强功能 | 17项 | 2项 | 1天 | ✅ 已完成 |
| **核心任务总计** | **84项** | **61项细项** | **6-8天** | **100%** (84/84) |
| 远期/待讨论 | 14个决策点 | ~40项条件任务 | 待定 | 待讨论 |

**说明:**
- 主要任务: checkbox 最外层项目 (- [ ])
- 子任务/细项: 带缩进的 bullet points (无 checkbox)
- 决策点: 需讨论确认是否实施的功能
- 预留测试: Phase 3 仅标记，Phase 4/6 对应功能完成时实现

---

## 🎯 执行原则

1. **严格按 Phase 顺序执行** - 每个 Phase 完成后才进入下一个
2. **Phase 完成标准** - 所有任务完成 + 测试通过 + CHANGELOG 记录
3. **依赖关系** - Phase N 依赖 Phase N-1，不可跳跃
4. **测试保护** - Phase 3 后所有修改必须有测试覆盖
5. **向后兼容** - Phase 5 迁移时保持旧代码可用（过渡期）
6. **文档同步** - 代码变更必须同步更新文档
7. **渐进式改进** - 保持每个 Phase 都有可工作版本

---

## ✅ 已完成 (可验证)

- [x] 实现核心教学模块 (deep_learning_fundamentals/cnn/rnn/advanced 等 9 个文件)
- [x] 添加中文注释和详细 docstring
- [x] 完成 DEEP_LEARNING_GUIDE.md 初稿

---

## 📝 快速启动

**Day 1 (Phase 1):**
创建 .gitignore, LICENSE, requirements, pyproject.toml, Makefile, README.md

**Day 2 (Phase 2):**
设计 ARCHITECTURE.md, 创建 main.py, 更新配套文档

**Day 3 (Phase 3):**
搭建 pytest 框架, 编写 smoke tests 和最小单元测试

**Week 2 (Phase 4-5):**
提取公共 utils, 迁移代码到新结构, 确保测试通过

**Week 3 (Phase 6-7):**
创建 examples, 重构 exercises, 添加可视化和数据集支持

---

---

## 🔄 变更日志

| 日期 | 变更 | 影响 Phase |
|------|------|-----------|
| 2025-12-01 | 完成: M3 国际化 (i18n/README_en.md + 关键 docstring 英文化) | M3 |
| 2025-12-01 | 完成: M2 Notebook 支持 (notebooks/ 目录 + 3 个示例 .ipynb) | M2 |
| 2025-12-01 | 完成: M1 交互式功能（LearningProgress/Quiz/HintSystem + 测试） | M1 |
| 2025-12-01 | 完成: M4 打包脚手架 (setup.py + MANIFEST.in + console_scripts dl-tutorial) | M4 |
| 2025-12-01 | 完成: M5 CI/CD (GitHub Actions pytest 工作流：Python 3.8/3.11) | M5 |
| 2025-12-01 | 完成: Phase 7 性能优化（矩阵乘法与卷积关键循环优化 + 补充测试 + 进度表校准至 84/84） | Phase 7 |
| 2024-11-26 | 完成: Phase 4 代码重构 (utils包+重构3个核心模块, 减少45行重复代码) | Phase 4 |
| 2024-11-26 | 完成: Phase 4 公共工具提取 (4个模块 + 50个新测试, 83个测试全部通过) | Phase 4 |
| 2024-11-26 | 完成: Phase 3 测试框架 (33个测试全部通过) | Phase 3 |
| 2024-11-25 | 修复: main.py 模块加载 bug + 添加模块 docstring (PR #3) | Phase 2 |
| 2024-11-25 | 去重: 移除 Phase 7 中重复的 RNN 梯度裁剪任务 | Phase 5, 7 |
| 2024-11-25 | 对齐: Phase 3 smoke test 改为测试"当前包结构" | Phase 3 |
| 2024-11-25 | 补充: Phase 3 预留 Phase 4-6 新功能的测试任务 | Phase 3, 6 |
| 2024-11-25 | 完善: 远期决策项添加"如是/否→任务"映射 | 远期 |
| 2024-11-25 | 修正: 更新任务统计表，区分主任务和子任务 | 全局 |
| 2024-11-25 | 去硬编码: 移除"9个模块"等硬编码数字，改为动态发现 | Phase 2, 3, M2 |
| 2024-11-25 | 明确测试时机: Phase 3 预留测试仅标记，Phase 4/6 实现 | Phase 3 |
| 2024-11-25 | 精确统计: 重新统计所有 Phase 的主任务和子任务数量 | 全局 |
| 2024-11-25 | 最终修正: 实际统计 checkbox 数量 (Phase 1=10, 总计83项) | 全局 |

---

**最后更新:** 2025-12-01
**当前 Phase:** Phase 7 已完成 ✓
**总体进度:** 100% (84/84项核心任务已完成)
