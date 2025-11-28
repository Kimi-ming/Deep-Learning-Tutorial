# -*- coding: utf-8 -*-
"""
深度学习基础教程 - 从神经网络到深度学习

包含：感知机、多层感知机、激活函数、损失函数、优化算法等基础概念。

注意: 此文件已作为兼容入口，推荐使用 `from deep_learning.fundamentals import MLP, DeepNetwork`
"""

import random
import math
import json
import warnings

# 导入 utils 工具函数
from deep_learning.utils import (
    relu, relu_derivative,
    leaky_relu,
    he_normal
)

# 新包迁移引用
from deep_learning.fundamentals.deep_network import DeepNetwork
from deep_learning.fundamentals.perceptron import Perceptron

# 兼容提示
warnings.warn(
    "deep_learning_fundamentals.py 将迁移到 deep_learning/fundamentals/ 包，"
    "请使用 from deep_learning.fundamentals import MLP, DeepNetwork",
    DeprecationWarning,
    stacklevel=2,
)

def deep_learning_introduction():
    """
    深度学习入门概念
    
    深度学习是机器学习的一个分支，使用多层神经网络来学习数据的抽象表示。
    
    核心概念：
    - 深度：网络有多个隐藏层（通常3层以上称为深度网络）
    - 表示学习：自动学习有用的特征表示
    - 层次化特征：低层学习简单特征，高层学习复杂特征
    - 端到端学习：从原始数据直接学习到最终输出
    
    深度学习 vs 传统机器学习：
    1. 特征工程：传统ML需要手工设计特征，DL自动学习特征
    2. 数据需求：DL通常需要更多数据
    3. 计算需求：DL需要更多计算资源
    4. 性能：在大数据集上，DL通常性能更好
    
    深度学习发展历程：
    - 1950s: 感知机的诞生
    - 1980s: 反向传播算法
    - 2006: 深度信念网络，深度学习复兴
    - 2012: AlexNet在ImageNet获得突破
    - 2010s至今: CNN、RNN、Transformer等架构发展
    """
    print("=== 深度学习基础概念 ===")
    print("深度学习 = 多层神经网络 + 大数据 + 强计算力")
    print()
    
    print("深度学习的三大要素：")
    print("1. 算法：神经网络架构和训练方法")
    print("2. 数据：大规模标注数据集")
    print("3. 算力：GPU并行计算能力")
    print()
    
    print("主要应用领域：")
    print("• 计算机视觉：图像分类、目标检测、图像生成")
    print("• 自然语言处理：机器翻译、文本生成、情感分析")
    print("• 语音处理：语音识别、语音合成")
    print("• 推荐系统：个性化推荐、广告投放")
    print("• 游戏AI：围棋、电子游戏AI")
    print()

def deep_learning_architectures_overview():
    """深度学习主要架构概览"""
    print("=== 深度学习主要架构 ===")
    
    architectures = {
        "多层感知机 (MLP)": {
            "description": "最基础的深度神经网络",
            "structure": "全连接层堆叠",
            "applications": ["表格数据分类", "简单回归问题"],
            "advantages": ["结构简单", "易于理解"],
            "disadvantages": ["参数过多", "难以处理空间结构"]
        },
        
        "卷积神经网络 (CNN)": {
            "description": "专门处理网格状数据（如图像）",
            "structure": "卷积层 + 池化层 + 全连接层",
            "applications": ["图像分类", "目标检测", "图像分割"],
            "advantages": ["局部连接", "权重共享", "平移不变性"],
            "disadvantages": ["主要用于图像", "需要大量数据"]
        },
        
        "循环神经网络 (RNN)": {
            "description": "处理序列数据的网络架构",
            "structure": "循环连接的隐藏状态",
            "applications": ["自然语言处理", "时间序列预测", "语音识别"],
            "advantages": ["处理变长序列", "记忆历史信息"],
            "disadvantages": ["梯度消失", "难以并行化"]
        },
        
        "长短期记忆网络 (LSTM)": {
            "description": "解决RNN梯度消失问题的改进版本",
            "structure": "门控机制 + 细胞状态",
            "applications": ["机器翻译", "文本生成", "情感分析"],
            "advantages": ["长期记忆能力", "缓解梯度消失"],
            "disadvantages": ["结构复杂", "计算开销大"]
        },
        
        "Transformer": {
            "description": "基于注意力机制的架构",
            "structure": "自注意力 + 前馈网络",
            "applications": ["机器翻译", "文本生成", "预训练语言模型"],
            "advantages": ["并行化", "长距离依赖", "可解释性"],
            "disadvantages": ["内存消耗大", "需要大量数据"]
        }
    }
    
    for name, info in architectures.items():
        print(f"\n【{name}】")
        print(f"描述: {info['description']}")
        print(f"结构: {info['structure']}")
        print(f"应用: {', '.join(info['applications'])}")
        print(f"优点: {', '.join(info['advantages'])}")
        print(f"缺点: {', '.join(info['disadvantages'])}")

def deep_network_challenges():
    """深度网络训练挑战"""
    print("\n=== 深度网络训练挑战 ===")
    
    challenges = {
        "梯度消失/爆炸": {
            "problem": "反向传播时梯度过小或过大",
            "causes": ["激活函数饱和", "权重初始化不当", "网络过深"],
            "solutions": ["使用ReLU激活函数", "批归一化", "残差连接", "梯度裁剪"]
        },
        
        "过拟合": {
            "problem": "模型在训练集上表现好，在测试集上表现差",
            "causes": ["模型复杂度过高", "训练数据不足", "训练时间过长"],
            "solutions": ["Dropout", "L1/L2正则化", "数据增强", "早停法"]
        },
        
        "内部协变量偏移": {
            "problem": "训练过程中每层输入分布发生变化",
            "causes": ["参数更新导致输入分布变化"],
            "solutions": ["批归一化", "层归一化", "组归一化"]
        },
        
        "训练速度慢": {
            "problem": "深度网络参数多，训练时间长",
            "causes": ["网络复杂度高", "数据量大", "计算资源限制"],
            "solutions": ["GPU并行", "分布式训练", "混合精度训练", "模型压缩"]
        }
    }
    
    for challenge, info in challenges.items():
        print(f"\n【{challenge}】")
        print(f"问题: {info['problem']}")
        print(f"原因: {', '.join(info['causes'])}")
        print(f"解决方案: {', '.join(info['solutions'])}")

def modern_training_techniques():
    """现代训练技巧"""
    print("\n=== 现代深度学习训练技巧 ===")
    
    techniques = {
        "优化算法": [
            "SGD + 动量: 加速收敛，减少震荡",
            "Adam: 自适应学习率，适用于大多数问题",
            "AdamW: Adam + 权重衰减，更好的正则化",
            "学习率调度: 余弦退火、分段衰减"
        ],
        
        "正则化技术": [
            "Dropout: 随机丢弃神经元，防止过拟合",
            "Batch Normalization: 归一化输入，加速训练",
            "Data Augmentation: 数据增强，增加数据多样性",
            "Early Stopping: 在验证集性能下降时停止训练"
        ],
        
        "网络设计": [
            "残差连接 (ResNet): 解决深度网络退化问题",
            "密集连接 (DenseNet): 特征重用，参数效率",
            "注意力机制: 动态关注重要信息",
            "跳跃连接: 连接不同层，信息流动更好"
        ],
        
        "训练策略": [
            "预训练 + 微调: 利用预训练模型，减少训练时间",
            "迁移学习: 将知识从一个任务迁移到另一个任务",
            "多任务学习: 同时学习多个相关任务",
            "自监督学习: 从无标签数据中学习表示"
        ]
    }
    
    for category, items in techniques.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"• {item}")

class DeepNetwork:
    """
    深度神经网络实现
    支持多种现代技术
    """
    
    def __init__(self, layers, learning_rate=0.001, activation='relu', 
                 use_batch_norm=False, dropout_rate=0.0):
        """
        初始化深度网络
        
        参数:
        layers: 每层神经元数量
        learning_rate: 学习率
        activation: 激活函数
        use_batch_norm: 是否使用批归一化
        dropout_rate: Dropout比率
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # 使用He初始化（适合ReLU）
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            # 使用 utils.he_normal 进行权重初始化
            weights = he_normal((layers[i], layers[i + 1]))
            self.weights.append(weights)

            # 偏置初始化为0
            biases = [0.0 for _ in range(layers[i + 1])]
            self.biases.append(biases)
        
        # 批归一化参数
        if use_batch_norm:
            self.bn_gamma = []  # 缩放参数
            self.bn_beta = []   # 偏移参数
            self.bn_running_mean = []  # 运行时均值
            self.bn_running_var = []   # 运行时方差
            
            for i in range(len(layers) - 1):
                self.bn_gamma.append([1.0 for _ in range(layers[i + 1])])
                self.bn_beta.append([0.0 for _ in range(layers[i + 1])])
                self.bn_running_mean.append([0.0 for _ in range(layers[i + 1])])
                self.bn_running_var.append([1.0 for _ in range(layers[i + 1])])
        
        # 训练历史
        self.loss_history = []
        self.val_loss_history = []
        
        print(f"深度网络初始化完成:")
        print(f"结构: {' -> '.join(map(str, layers))}")
        print(f"激活函数: {activation}")
        print(f"批归一化: {'是' if use_batch_norm else '否'}")
        print(f"Dropout: {dropout_rate}")
        print(f"总参数量: {self.count_parameters()}")
    
    def count_parameters(self):
        """计算总参数数量"""
        total = 0
        for i in range(len(self.weights)):
            total += len(self.weights[i]) * len(self.weights[i][0])
            total += len(self.biases[i])
        
        if self.use_batch_norm:
            for i in range(len(self.bn_gamma)):
                total += len(self.bn_gamma[i]) * 2  # gamma和beta
        
        return total
    
    # 注意: 激活函数现在从 deep_learning.utils 导入
    # relu, leaky_relu 等函数已在模块顶部导入

    def batch_normalize(self, x, layer_idx, training=True, momentum=0.9, eps=1e-8):
        """批归一化"""
        if not self.use_batch_norm:
            return x
        
        if training:
            # 计算批统计量
            mean = sum(x) / len(x)
            var = sum((xi - mean) ** 2 for xi in x) / len(x)
            
            # 更新运行时统计量
            self.bn_running_mean[layer_idx] = [
                momentum * rm + (1 - momentum) * mean 
                for rm in self.bn_running_mean[layer_idx]
            ]
            self.bn_running_var[layer_idx] = [
                momentum * rv + (1 - momentum) * var 
                for rv in self.bn_running_var[layer_idx]
            ]
        else:
            # 使用运行时统计量
            mean = sum(self.bn_running_mean[layer_idx]) / len(self.bn_running_mean[layer_idx])
            var = sum(self.bn_running_var[layer_idx]) / len(self.bn_running_var[layer_idx])
        
        # 归一化
        x_norm = [(xi - mean) / math.sqrt(var + eps) for xi in x]
        
        # 缩放和偏移
        output = []
        for i, xi in enumerate(x_norm):
            if i < len(self.bn_gamma[layer_idx]):
                out = self.bn_gamma[layer_idx][i] * xi + self.bn_beta[layer_idx][i]
                output.append(out)
            else:
                output.append(xi)
        
        return output
    
    def dropout(self, x, training=True):
        """Dropout正则化"""
        if not training or self.dropout_rate == 0:
            return x
        
        # 随机丢弃神经元
        mask = [1 if random.random() > self.dropout_rate else 0 for _ in x]
        scale = 1.0 / (1.0 - self.dropout_rate)  # 缩放补偿
        
        return [xi * mi * scale for xi, mi in zip(x, mask)]
    
    def forward(self, inputs, training=True):
        """前向传播"""
        current = inputs
        activations = [inputs]
        z_values = []
        
        for i in range(len(self.weights)):
            # 线性变换
            z = []
            for j in range(len(self.weights[i])):
                weighted_sum = sum(w * inp for w, inp in zip(self.weights[i][j], current))
                weighted_sum += self.biases[i][j]
                z.append(weighted_sum)
            
            z_values.append(z)
            
            # 批归一化
            if self.use_batch_norm and i < len(self.weights) - 1:  # 不在输出层使用
                z = self.batch_normalize(z, i, training)
            
            # 激活函数
            if i < len(self.weights) - 1:  # 隐藏层
                if self.activation == 'relu':
                    activated = [relu(zi) for zi in z]
                elif self.activation == 'leaky_relu':
                    activated = [leaky_relu(zi) for zi in z]
                else:
                    activated = z  # 线性

                # Dropout
                activated = self.dropout(activated, training)
            else:  # 输出层
                activated = z  # 输出层通常不使用激活函数
            
            activations.append(activated)
            current = activated
        
        return activations, z_values
    
    def predict(self, inputs):
        """预测"""
        activations, _ = self.forward(inputs, training=False)
        return activations[-1]
    
    def train_batch(self, X_batch, y_batch):
        """训练一个批次"""
        total_loss = 0
        
        for i in range(len(X_batch)):
            inputs = X_batch[i]
            targets = y_batch[i] if isinstance(y_batch[i], list) else [y_batch[i]]
            
            # 前向传播
            activations, z_values = self.forward(inputs, training=True)
            predictions = activations[-1]
            
            # 计算损失
            loss = sum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(targets)
            total_loss += loss
            
            # 反向传播（简化版本）
            # 这里可以实现完整的反向传播算法
            # 为了简化，我们使用简单的梯度近似
            
        return total_loss / len(X_batch)

def transfer_learning_concept():
    """迁移学习概念"""
    print("\n=== 迁移学习 ===")
    
    print("迁移学习的核心思想：")
    print("利用在大数据集上预训练的模型，迁移到新的任务上")
    print()
    
    print("迁移学习的优势：")
    print("• 减少训练时间和计算资源需求")
    print("• 在小数据集上也能取得好效果")  
    print("• 利用预训练模型学到的通用特征")
    print("• 提高模型性能和泛化能力")
    print()
    
    print("迁移学习的方式：")
    print("1. 特征提取: 冻结预训练模型，只训练分类器")
    print("2. 微调: 对预训练模型进行细微调整")
    print("3. 端到端微调: 对整个网络进行微调")
    print()
    
    print("预训练模型示例：")
    print("• 计算机视觉: ResNet, VGG, EfficientNet")
    print("• 自然语言处理: BERT, GPT, T5")
    print("• 语音识别: Wav2Vec, Whisper")

def deep_learning_frameworks():
    """深度学习框架介绍"""
    print("\n=== 深度学习框架 ===")
    
    frameworks = {
        "TensorFlow/Keras": {
            "特点": ["Google开发", "工业级部署", "易于使用的高级API"],
            "优势": ["生态完善", "部署便利", "社区活跃"],
            "适用": ["工业应用", "初学者", "研究"]
        },
        
        "PyTorch": {
            "特点": ["Facebook开发", "动态计算图", "Pythonic设计"],
            "优势": ["灵活性高", "调试方便", "研究友好"],
            "适用": ["学术研究", "快速原型", "复杂模型"]
        },
        
        "JAX": {
            "特点": ["Google开发", "函数式编程", "XLA编译"],
            "优势": ["性能优秀", "自动微分", "数值计算"],
            "适用": ["高性能计算", "科学计算", "研究"]
        }
    }
    
    for name, info in frameworks.items():
        print(f"\n【{name}】")
        print(f"特点: {', '.join(info['特点'])}")
        print(f"优势: {', '.join(info['优势'])}")
        print(f"适用: {', '.join(info['适用'])}")

def learning_roadmap():
    """深度学习学习路线图"""
    print("\n=== 深度学习学习路线图 ===")
    
    roadmap = {
        "基础阶段 (1-2个月)": [
            "理解神经网络基本原理",
            "掌握反向传播算法",
            "熟悉激活函数和损失函数",
            "学习基本的正则化技术",
            "实现简单的多层感知机"
        ],
        
        "进阶阶段 (2-3个月)": [
            "学习卷积神经网络(CNN)",
            "理解循环神经网络(RNN/LSTM)",
            "掌握现代训练技巧",
            "学习使用深度学习框架",
            "完成图像分类和文本分类项目"
        ],
        
        "高级阶段 (3-4个月)": [
            "学习Transformer架构",
            "理解注意力机制",
            "掌握预训练和微调",
            "学习生成对抗网络(GAN)",
            "完成复杂的端到端项目"
        ],
        
        "专家阶段 (持续学习)": [
            "跟踪最新研究进展",
            "学习特定领域的先进技术",
            "参与开源项目贡献",
            "发表学术论文或技术博客",
            "解决实际工业问题"
        ]
    }
    
    for stage, tasks in roadmap.items():
        print(f"\n【{stage}】")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")

def practical_tips():
    """实用技巧和建议"""
    print("\n=== 实用技巧和建议 ===")
    
    print("编程实践技巧：")
    print("• 从简单模型开始，逐步增加复杂度")
    print("• 使用预训练模型作为起点")
    print("• 重视数据质量和预处理")
    print("• 建立完善的实验记录系统")
    print("• 多做消融实验分析模型性能")
    print()
    
    print("调试技巧：")
    print("• 检查数据加载和预处理流程")
    print("• 验证模型输出形状和数值范围")
    print("• 监控梯度大小和分布")
    print("• 可视化中间层特征图")
    print("• 对比简化版本的模型性能")
    print()
    
    print("性能优化：")
    print("• 使用混合精度训练")
    print("• 优化数据加载流水线")
    print("• 合理设置批大小")
    print("• 使用分布式训练")
    print("• 考虑模型压缩和量化")

def main():
    """主函数"""
    print("🚀 深度学习基础教程")
    print("=" * 50)
    
    deep_learning_introduction()
    deep_learning_architectures_overview()
    deep_network_challenges()
    modern_training_techniques()
    transfer_learning_concept()
    deep_learning_frameworks()
    learning_roadmap()
    practical_tips()
    
    print("\n" + "=" * 50)
    print("🎯 总结")
    print("深度学习是机器学习的重要分支：")
    print("• 使用多层神经网络学习复杂模式")
    print("• 在大数据集上表现优异")
    print("• 需要掌握现代训练技巧")
    print("• 广泛应用于各个领域")
    print()
    print("学习建议：")
    print("• 理论与实践并重")
    print("• 多做项目巩固知识")
    print("• 关注最新研究进展")
    print("• 参与开源社区")
    print()
    print("下一步：选择具体方向深入学习！")
    print("推荐：先学习CNN处理图像，或学习Transformer处理文本")

if __name__ == "__main__":
    main()
