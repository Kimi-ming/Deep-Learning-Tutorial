# -*- coding: utf-8 -*-
"""
前沿深度学习架构 - Transformer、ViT、EfficientNet、MoE

包含：Transformer详细实现、Vision Transformer、EfficientNet、混合专家模型等前沿架构。
"""
# Cutting-Edge Deep Learning Architectures: 最新架构实现与原理

import random
import math
import json

def cutting_edge_intro():
    """前沿架构介绍"""
    print("=== 前沿深度学习架构 ===")
    print("探索最新的神经网络架构和技术")
    print()
    print("前沿架构:")
    print("• Vision Transformer (ViT)")
    print("• BERT与GPT系列模型")
    print("• EfficientNet与神经架构搜索")
    print("• 残差网络的进化 (ResNeXt, DenseNet)")
    print("• 注意力机制的变种 (Self-Attention, Cross-Attention)")
    print("• 混合专家模型 (Mixture of Experts)")
    print("• 神经ODE与连续深度模型")
    print()

def transformer_detailed_implementation():
    """Transformer详细实现"""
    print("\n" + "="*70)
    print("Transformer架构深度实现")
    print("="*70)
    
    print("Transformer核心组件:")
    print("• Multi-Head Self-Attention")
    print("• Position Encoding")
    print("• Feed-Forward Networks")
    print("• Layer Normalization")
    print("• Residual Connections")
    print()
    
    class MultiHeadAttention:
        """多头注意力机制实现"""
        
        def __init__(self, d_model=512, num_heads=8):
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # 初始化权重矩阵
            self.W_q = self._init_weights(d_model, d_model)
            self.W_k = self._init_weights(d_model, d_model)
            self.W_v = self._init_weights(d_model, d_model)
            self.W_o = self._init_weights(d_model, d_model)
            
        def _init_weights(self, in_dim, out_dim):
            """Xavier初始化"""
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            return [[random.uniform(-limit, limit) for _ in range(out_dim)] 
                   for _ in range(in_dim)]
        
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            """缩放点积注意力"""
            # 计算注意力分数 scores = Q @ K.T / sqrt(d_k)
            seq_len = len(Q)
            scores = []
            
            for i in range(seq_len):
                score_row = []
                for j in range(seq_len):
                    score = sum(Q[i][k] * K[j][k] for k in range(self.d_k))
                    score /= math.sqrt(self.d_k)
                    score_row.append(score)
                scores.append(score_row)
            
            # 应用mask (如果有)
            if mask is not None:
                for i in range(seq_len):
                    for j in range(seq_len):
                        if mask[i][j] == 0:
                            scores[i][j] = float('-inf')
            
            # Softmax
            attention_weights = []
            for i in range(seq_len):
                # 数值稳定的softmax
                max_score = max(scores[i])
                exp_scores = [math.exp(s - max_score) for s in scores[i]]
                sum_exp = sum(exp_scores)
                weights = [exp_s / sum_exp for exp_s in exp_scores]
                attention_weights.append(weights)
            
            # 计算输出 output = attention_weights @ V
            output = []
            for i in range(seq_len):
                output_vector = [0.0] * self.d_k
                for j in range(seq_len):
                    for k in range(self.d_k):
                        output_vector[k] += attention_weights[i][j] * V[j][k]
                output.append(output_vector)
            
            return output, attention_weights
        
        def forward(self, x, mask=None):
            """多头注意力前向传播"""
            seq_len = len(x)
            
            # 线性变换得到Q, K, V
            Q = self._linear_transform(x, self.W_q)
            K = self._linear_transform(x, self.W_k)
            V = self._linear_transform(x, self.W_v)
            
            # 分割为多个头
            Q_heads = self._split_heads(Q)
            K_heads = self._split_heads(K)
            V_heads = self._split_heads(V)
            
            # 对每个头计算注意力
            attention_outputs = []
            all_attention_weights = []
            
            for h in range(self.num_heads):
                output, weights = self.scaled_dot_product_attention(
                    Q_heads[h], K_heads[h], V_heads[h], mask)
                attention_outputs.append(output)
                all_attention_weights.append(weights)
            
            # 连接所有头的输出
            concatenated = self._concat_heads(attention_outputs)
            
            # 最终线性变换
            final_output = self._linear_transform(concatenated, self.W_o)
            
            return final_output, all_attention_weights
        
        def _linear_transform(self, x, W):
            """线性变换 x @ W"""
            seq_len = len(x)
            d_in = len(x[0])
            d_out = len(W[0])
            
            output = []
            for i in range(seq_len):
                output_vector = []
                for j in range(d_out):
                    value = sum(x[i][k] * W[k][j] for k in range(d_in))
                    output_vector.append(value)
                output.append(output_vector)
            
            return output
        
        def _split_heads(self, x):
            """将输入分割为多个头"""
            seq_len = len(x)
            heads = []
            
            for h in range(self.num_heads):
                head_data = []
                for i in range(seq_len):
                    head_vector = x[i][h * self.d_k:(h + 1) * self.d_k]
                    head_data.append(head_vector)
                heads.append(head_data)
            
            return heads
        
        def _concat_heads(self, heads):
            """连接多个头的输出"""
            seq_len = len(heads[0])
            concatenated = []
            
            for i in range(seq_len):
                concat_vector = []
                for h in range(self.num_heads):
                    concat_vector.extend(heads[h][i])
                concatenated.append(concat_vector)
            
            return concatenated
    
    class PositionalEncoding:
        """位置编码"""
        
        def __init__(self, d_model=512, max_len=5000):
            self.d_model = d_model
            self.max_len = max_len
            self.encoding = self._generate_encoding()
        
        def _generate_encoding(self):
            """生成正弦位置编码"""
            encoding = []
            
            for pos in range(self.max_len):
                pos_encoding = []
                for i in range(self.d_model):
                    if i % 2 == 0:
                        # 偶数位置使用sin
                        angle = pos / (10000 ** (i / self.d_model))
                        pos_encoding.append(math.sin(angle))
                    else:
                        # 奇数位置使用cos
                        angle = pos / (10000 ** ((i-1) / self.d_model))
                        pos_encoding.append(math.cos(angle))
                encoding.append(pos_encoding)
            
            return encoding
        
        def add_positional_encoding(self, x):
            """添加位置编码到输入"""
            seq_len = len(x)
            encoded = []
            
            for i in range(seq_len):
                encoded_vector = []
                for j in range(len(x[i])):
                    encoded_value = x[i][j] + self.encoding[i][j]
                    encoded_vector.append(encoded_value)
                encoded.append(encoded_vector)
            
            return encoded
    
    class TransformerBlock:
        """Transformer块"""
        
        def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
            self.d_model = d_model
            self.d_ff = d_ff
            self.dropout = dropout
            
            self.attention = MultiHeadAttention(d_model, num_heads)
            self.pos_encoding = PositionalEncoding(d_model)
            
            # 前馈网络权重
            self.W1 = self._init_weights(d_model, d_ff)
            self.b1 = [0.0] * d_ff
            self.W2 = self._init_weights(d_ff, d_model)
            self.b2 = [0.0] * d_model
            
        def _init_weights(self, in_dim, out_dim):
            """Xavier初始化"""
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            return [[random.uniform(-limit, limit) for _ in range(out_dim)] 
                   for _ in range(in_dim)]
        
        def layer_norm(self, x, eps=1e-6):
            """层归一化"""
            normalized = []
            
            for vector in x:
                # 计算均值和方差
                mean = sum(vector) / len(vector)
                variance = sum((v - mean) ** 2 for v in vector) / len(vector)
                
                # 归一化
                norm_vector = [(v - mean) / math.sqrt(variance + eps) for v in vector]
                normalized.append(norm_vector)
            
            return normalized
        
        def feed_forward(self, x):
            """前馈网络"""
            # 第一层: ReLU(x @ W1 + b1)
            hidden = []
            for vector in x:
                hidden_vector = []
                for i in range(self.d_ff):
                    value = sum(vector[j] * self.W1[j][i] for j in range(len(vector))) + self.b1[i]
                    hidden_vector.append(max(0, value))  # ReLU
                hidden.append(hidden_vector)
            
            # 第二层: hidden @ W2 + b2
            output = []
            for vector in hidden:
                output_vector = []
                for i in range(self.d_model):
                    value = sum(vector[j] * self.W2[j][i] for j in range(len(vector))) + self.b2[i]
                    output_vector.append(value)
                output.append(output_vector)
            
            return output
        
        def forward(self, x, mask=None):
            """Transformer块前向传播"""
            # 添加位置编码
            x_pos = self.pos_encoding.add_positional_encoding(x)
            
            # Multi-Head Self-Attention + 残差连接 + 层归一化
            attn_output, attn_weights = self.attention.forward(x_pos, mask)
            
            # 残差连接
            x1 = []
            for i in range(len(x_pos)):
                residual_vector = []
                for j in range(len(x_pos[i])):
                    residual_vector.append(x_pos[i][j] + attn_output[i][j])
                x1.append(residual_vector)
            
            # 层归一化
            x1_norm = self.layer_norm(x1)
            
            # Feed-Forward + 残差连接 + 层归一化
            ff_output = self.feed_forward(x1_norm)
            
            # 残差连接
            x2 = []
            for i in range(len(x1_norm)):
                residual_vector = []
                for j in range(len(x1_norm[i])):
                    residual_vector.append(x1_norm[i][j] + ff_output[i][j])
                x2.append(residual_vector)
            
            # 层归一化
            output = self.layer_norm(x2)
            
            return output, attn_weights
    
    # Transformer演示
    print("Transformer架构演示:")
    
    # 创建模拟输入 (seq_len=4, d_model=8 为了演示)
    seq_len = 4
    d_model = 8
    
    # 随机输入序列
    input_sequence = []
    for i in range(seq_len):
        vector = [random.uniform(-1, 1) for _ in range(d_model)]
        input_sequence.append(vector)
    
    print(f"输入序列形状: ({seq_len}, {d_model})")
    print(f"前3个位置的输入:")
    for i in range(min(3, seq_len)):
        formatted_vector = [f"{v:.3f}" for v in input_sequence[i]]
        print(f"  位置{i}: [{', '.join(formatted_vector)}]")
    
    # 创建Transformer块
    transformer = TransformerBlock(d_model=d_model, num_heads=2, d_ff=16)
    
    # 前向传播
    output, attention_weights = transformer.forward(input_sequence)
    
    print(f"\nTransformer输出:")
    for i in range(min(3, seq_len)):
        formatted_output = [f"{v:.3f}" for v in output[i]]
        print(f"  位置{i}: [{', '.join(formatted_output)}]")
    
    print(f"\n注意力权重矩阵 (头1):")
    for i in range(seq_len):
        weights_str = [f"{w:.3f}" for w in attention_weights[0][i]]
        print(f"  位置{i}: [{', '.join(weights_str)}]")
    
    print(f"\nTransformer关键特性:")
    print(f"• 自注意力机制：每个位置关注所有位置")
    print(f"• 位置编码：为序列添加位置信息")
    print(f"• 残差连接：缓解梯度消失问题")
    print(f"• 层归一化：稳定训练过程")

def vision_transformer_implementation():
    """Vision Transformer实现"""
    print("\n" + "="*70)
    print("Vision Transformer (ViT)")
    print("="*70)
    
    print("ViT核心思想:")
    print("• 将图像分割为patches")
    print("• 每个patch视为序列中的token")
    print("• 应用标准Transformer架构")
    print("• 添加可学习的分类token")
    print()
    
    class VisionTransformer:
        """Vision Transformer实现"""
        
        def __init__(self, image_size=224, patch_size=16, num_classes=1000, 
                     d_model=768, num_heads=12, num_layers=12):
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_classes = num_classes
            self.d_model = d_model
            self.num_heads = num_heads
            self.num_layers = num_layers
            
            # 计算patch数量
            self.num_patches = (image_size // patch_size) ** 2
            self.seq_len = self.num_patches + 1  # +1 for class token
            
            # 初始化参数
            self.patch_embedding = self._init_patch_embedding()
            self.class_token = [random.gauss(0, 0.02) for _ in range(d_model)]
            self.position_embeddings = self._init_position_embeddings()
            
            print(f"ViT配置:")
            print(f"  图像大小: {image_size}x{image_size}")
            print(f"  Patch大小: {patch_size}x{patch_size}")
            print(f"  Patch数量: {self.num_patches}")
            print(f"  序列长度: {self.seq_len} (包含class token)")
            print(f"  模型维度: {d_model}")
        
        def _init_patch_embedding(self):
            """初始化patch嵌入权重"""
            input_dim = self.patch_size * self.patch_size * 3  # RGB channels
            limit = math.sqrt(6.0 / (input_dim + self.d_model))
            return [[random.uniform(-limit, limit) for _ in range(self.d_model)] 
                   for _ in range(input_dim)]
        
        def _init_position_embeddings(self):
            """初始化位置嵌入"""
            embeddings = []
            for i in range(self.seq_len):
                embedding = [random.gauss(0, 0.02) for _ in range(self.d_model)]
                embeddings.append(embedding)
            return embeddings
        
        def image_to_patches(self, image):
            """将图像转换为patches"""
            # 模拟图像切分过程
            patches = []
            
            for i in range(0, self.image_size, self.patch_size):
                for j in range(0, self.image_size, self.patch_size):
                    # 提取patch (简化为随机值)
                    patch = []
                    for c in range(3):  # RGB
                        for pi in range(self.patch_size):
                            for pj in range(self.patch_size):
                                # 在实际实现中，这里应该是 image[i+pi][j+pj][c]
                                patch.append(random.uniform(0, 1))
                    patches.append(patch)
            
            return patches
        
        def patch_embedding_forward(self, patches):
            """Patch嵌入"""
            embedded_patches = []
            
            for patch in patches:
                embedded = []
                for i in range(self.d_model):
                    value = sum(patch[j] * self.patch_embedding[j][i] 
                              for j in range(len(patch)))
                    embedded.append(value)
                embedded_patches.append(embedded)
            
            return embedded_patches
        
        def add_class_token_and_position(self, embedded_patches):
            """添加class token和位置嵌入"""
            # 添加class token到序列开头
            sequence = [self.class_token[:]] + embedded_patches
            
            # 添加位置嵌入
            for i in range(len(sequence)):
                for j in range(self.d_model):
                    sequence[i][j] += self.position_embeddings[i][j]
            
            return sequence
        
        def forward(self, image):
            """ViT前向传播"""
            # 1. 图像到patches
            patches = self.image_to_patches(image)
            
            # 2. Patch嵌入
            embedded_patches = self.patch_embedding_forward(patches)
            
            # 3. 添加class token和位置嵌入
            sequence = self.add_class_token_and_position(embedded_patches)
            
            # 4. Transformer编码器 (简化为单层)
            transformer = TransformerBlock(self.d_model, self.num_heads, self.d_model * 4)
            encoded_sequence, attention_weights = transformer.forward(sequence)
            
            # 5. 提取class token的输出用于分类
            class_output = encoded_sequence[0]
            
            # 6. 分类头 (简化为线性层)
            # 在实际实现中，这里是一个全连接层
            classification_score = sum(class_output) / len(class_output)  # 简化
            
            return classification_score, attention_weights
    
    # ViT演示
    print("Vision Transformer演示:")
    
    # 创建ViT模型 (小尺寸用于演示)
    vit = VisionTransformer(image_size=32, patch_size=8, d_model=64, num_heads=4)
    
    # 模拟输入图像
    dummy_image = "dummy_image"  # 在实际中是3D数组
    
    # 前向传播
    classification_score, attention_weights = vit.forward(dummy_image)
    
    print(f"\n分类输出: {classification_score:.6f}")
    
    print(f"\nViT相比CNN的优势:")
    print(f"• 长距离依赖：自注意力可以捕获全局信息")
    print(f"• 可解释性：注意力权重提供可视化")
    print(f"• 可扩展性：容易扩展到大模型")
    print(f"• 预训练：可以在大数据集上预训练")

def efficient_neural_architecture():
    """高效神经网络架构"""
    print("\n" + "="*70)
    print("EfficientNet与神经架构搜索")
    print("="*70)
    
    print("EfficientNet核心创新:")
    print("• 复合缩放法则：平衡深度、宽度、分辨率")
    print("• MBConv块：移动倒置残差块")
    print("• Squeeze-and-Excitation：通道注意力")
    print("• 神经架构搜索(NAS)：自动设计网络")
    print()
    
    class MBConvBlock:
        """Mobile Inverted Bottleneck Convolution块"""
        
        def __init__(self, input_channels, output_channels, expansion_ratio=6, 
                     kernel_size=3, stride=1, se_ratio=0.25):
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.expansion_ratio = expansion_ratio
            self.kernel_size = kernel_size
            self.stride = stride
            self.se_ratio = se_ratio
            
            # 计算中间通道数
            self.expanded_channels = input_channels * expansion_ratio
            
            # 初始化权重 (简化表示)
            self.expand_conv = self._init_conv_weights(input_channels, self.expanded_channels, 1)
            self.depthwise_conv = self._init_conv_weights(self.expanded_channels, self.expanded_channels, kernel_size)
            self.se_weights = self._init_se_weights()
            self.project_conv = self._init_conv_weights(self.expanded_channels, output_channels, 1)
            
        def _init_conv_weights(self, in_ch, out_ch, kernel_size):
            """初始化卷积权重"""
            fan_out = out_ch * kernel_size * kernel_size
            std = math.sqrt(2.0 / fan_out)
            return {
                'weight': random.gauss(0, std),
                'bias': 0.0,
                'in_channels': in_ch,
                'out_channels': out_ch,
                'kernel_size': kernel_size
            }
        
        def _init_se_weights(self):
            """初始化Squeeze-and-Excitation权重"""
            se_channels = max(1, int(self.input_channels * self.se_ratio))
            return {
                'fc1': self._init_conv_weights(self.expanded_channels, se_channels, 1),
                'fc2': self._init_conv_weights(se_channels, self.expanded_channels, 1)
            }
        
        def squeeze_and_excitation(self, x):
            """Squeeze-and-Excitation模块"""
            batch_size = len(x)
            channels = len(x[0])
            
            # Global Average Pooling (Squeeze)
            se_input = []
            for b in range(batch_size):
                channel_means = []
                for c in range(channels):
                    mean_val = sum(x[b][c]) / len(x[b][c])  # 简化的GAP
                    channel_means.append(mean_val)
                se_input.append(channel_means)
            
            # Excitation: FC -> ReLU -> FC -> Sigmoid
            se_output = []
            for b in range(batch_size):
                # 第一个FC层 + ReLU
                fc1_out = []
                se_channels = self.se_weights['fc1']['out_channels']
                for i in range(se_channels):
                    val = sum(se_input[b][j] * random.uniform(-0.1, 0.1) 
                             for j in range(len(se_input[b])))
                    fc1_out.append(max(0, val))  # ReLU
                
                # 第二个FC层 + Sigmoid
                fc2_out = []
                for i in range(channels):
                    val = sum(fc1_out[j] * random.uniform(-0.1, 0.1) 
                             for j in range(len(fc1_out)))
                    fc2_out.append(1.0 / (1.0 + math.exp(-val)))  # Sigmoid
                
                se_output.append(fc2_out)
            
            # 应用注意力权重
            attended = []
            for b in range(batch_size):
                attended_channels = []
                for c in range(channels):
                    attended_channel = [val * se_output[b][c] for val in x[b][c]]
                    attended_channels.append(attended_channel)
                attended.append(attended_channels)
            
            return attended
        
        def forward(self, x):
            """MBConv块前向传播"""
            batch_size = len(x)
            
            print(f"  MBConv块处理:")
            print(f"    输入通道: {self.input_channels}")
            print(f"    扩展到: {self.expanded_channels} (扩展比率: {self.expansion_ratio})")
            
            # 1. Expansion (如果expansion_ratio > 1)
            if self.expansion_ratio != 1:
                # 1x1 卷积扩展通道
                expanded = x  # 简化实现
                print(f"    1x1扩展卷积: {self.input_channels} -> {self.expanded_channels}")
            else:
                expanded = x
            
            # 2. Depthwise Convolution
            # 深度可分离卷积 (简化实现)
            depthwise_out = expanded
            print(f"    深度卷积: kernel_size={self.kernel_size}, stride={self.stride}")
            
            # 3. Squeeze-and-Excitation
            if self.se_ratio > 0:
                se_out = self.squeeze_and_excitation(depthwise_out)
                print(f"    SE注意力: ratio={self.se_ratio}")
            else:
                se_out = depthwise_out
            
            # 4. Projection
            # 1x1 卷积投影到输出通道
            projected = se_out  # 简化实现
            print(f"    1x1投影: {self.expanded_channels} -> {self.output_channels}")
            
            # 5. Residual Connection (如果输入输出形状相同)
            if (self.input_channels == self.output_channels and 
                self.stride == 1 and len(x) == len(projected)):
                
                # 残差连接
                output = []
                for b in range(batch_size):
                    residual_channels = []
                    for c in range(min(len(x[b]), len(projected[b]))):
                        residual_channel = []
                        for i in range(min(len(x[b][c]), len(projected[b][c]))):
                            residual_channel.append(x[b][c][i] + projected[b][c][i])
                        residual_channels.append(residual_channel)
                    output.append(residual_channels)
                
                print(f"    残差连接: ✓")
            else:
                output = projected
                print(f"    残差连接: ✗ (形状不匹配)")
            
            return output
    
    class CompoundScaling:
        """复合缩放策略"""
        
        def __init__(self, phi=1.0):
            """
            phi: 复合系数
            depth_multiplier = α^phi
            width_multiplier = β^phi  
            resolution_multiplier = γ^phi
            约束: α * β^2 * γ^2 ≈ 2
            """
            self.phi = phi
            self.alpha = 1.2  # 深度缩放因子
            self.beta = 1.1   # 宽度缩放因子
            self.gamma = 1.15 # 分辨率缩放因子
            
        def scale_network(self, base_depth, base_width, base_resolution):
            """按复合缩放法则缩放网络"""
            depth_multiplier = self.alpha ** self.phi
            width_multiplier = self.beta ** self.phi
            resolution_multiplier = self.gamma ** self.phi
            
            scaled_depth = int(base_depth * depth_multiplier)
            scaled_width = int(base_width * width_multiplier)
            scaled_resolution = int(base_resolution * resolution_multiplier)
            
            # 验证约束条件
            constraint_value = depth_multiplier * (width_multiplier ** 2) * (resolution_multiplier ** 2)
            target_flops = 2 ** self.phi
            
            print(f"复合缩放 (φ={self.phi}):")
            print(f"  深度: {base_depth} -> {scaled_depth} (×{depth_multiplier:.2f})")
            print(f"  宽度: {base_width} -> {scaled_width} (×{width_multiplier:.2f})")
            print(f"  分辨率: {base_resolution} -> {scaled_resolution} (×{resolution_multiplier:.2f})")
            print(f"  约束检查: α×β²×γ² = {constraint_value:.2f} ≈ {target_flops:.2f}")
            
            return scaled_depth, scaled_width, scaled_resolution
    
    # EfficientNet演示
    print("EfficientNet架构演示:")
    
    # 1. MBConv块演示
    print("\n1. MBConv块演示:")
    mbconv = MBConvBlock(input_channels=32, output_channels=64, expansion_ratio=6)
    
    # 模拟输入 (batch_size=1, channels=32, spatial_dims简化)
    dummy_input = [[[random.uniform(0, 1) for _ in range(10)] for _ in range(32)]]
    
    mbconv_output = mbconv.forward(dummy_input)
    
    # 2. 复合缩放演示
    print(f"\n2. 复合缩放演示:")
    base_config = {
        'depth': 16,      # 基础层数
        'width': 64,      # 基础通道数
        'resolution': 224 # 基础分辨率
    }
    
    for phi in [0, 1, 2, 3]:
        scaler = CompoundScaling(phi=phi)
        scaled_depth, scaled_width, scaled_resolution = scaler.scale_network(
            base_config['depth'], base_config['width'], base_config['resolution'])
        print()
    
    print(f"\nEfficientNet的优势:")
    print(f"• 参数效率：相同精度下参数更少")
    print(f"• 计算效率：FLOPs更低")
    print(f"• 系统化缩放：避免手工调参")
    print(f"• 迁移性好：容易适配不同任务")

def mixture_of_experts():
    """混合专家模型"""
    print("\n" + "="*70)
    print("混合专家模型 (Mixture of Experts)")
    print("="*70)
    
    print("MoE核心思想:")
    print("• 稀疏激活：只激活部分专家")
    print("• 门控网络：学习如何选择专家")
    print("• 专家网络：专门处理特定类型的输入")
    print("• 可扩展性：增加专家而不增加计算")
    print()
    
    class MixtureOfExperts:
        """混合专家模型实现"""
        
        def __init__(self, input_dim, output_dim, num_experts=8, top_k=2):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_experts = num_experts
            self.top_k = top_k
            
            # 初始化门控网络
            self.gate_weights = self._init_weights(input_dim, num_experts)
            self.gate_bias = [0.0] * num_experts
            
            # 初始化专家网络
            self.experts = []
            for i in range(num_experts):
                expert = {
                    'weights': self._init_weights(input_dim, output_dim),
                    'bias': [random.uniform(-0.1, 0.1) for _ in range(output_dim)]
                }
                self.experts.append(expert)
            
            print(f"MoE配置:")
            print(f"  专家数量: {num_experts}")
            print(f"  Top-K: {top_k}")
            print(f"  输入维度: {input_dim}")
            print(f"  输出维度: {output_dim}")
        
        def _init_weights(self, in_dim, out_dim):
            """Xavier初始化"""
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            return [[random.uniform(-limit, limit) for _ in range(out_dim)] 
                   for _ in range(in_dim)]
        
        def gating_network(self, x):
            """门控网络计算专家权重"""
            # 计算门控分数 g = softmax(x @ W_g + b_g)
            gate_scores = []
            for i in range(self.num_experts):
                score = sum(x[j] * self.gate_weights[j][i] for j in range(len(x)))
                score += self.gate_bias[i]
                gate_scores.append(score)
            
            # Softmax归一化
            max_score = max(gate_scores)
            exp_scores = [math.exp(s - max_score) for s in gate_scores]
            sum_exp = sum(exp_scores)
            gate_weights = [exp_s / sum_exp for exp_s in exp_scores]
            
            return gate_weights
        
        def select_top_k_experts(self, gate_weights):
            """选择Top-K专家"""
            # 获取权重和索引的配对
            expert_pairs = [(weight, idx) for idx, weight in enumerate(gate_weights)]
            
            # 按权重降序排序
            expert_pairs.sort(reverse=True)
            
            # 选择Top-K
            top_k_experts = expert_pairs[:self.top_k]
            
            # 重新归一化Top-K权重
            top_k_weights = [pair[0] for pair in top_k_experts]
            weight_sum = sum(top_k_weights)
            
            if weight_sum > 0:
                normalized_weights = [w / weight_sum for w in top_k_weights]
            else:
                normalized_weights = [1.0 / self.top_k] * self.top_k
            
            # 返回选中的专家索引和归一化权重
            selected_experts = [(pair[1], normalized_weights[i]) 
                              for i, pair in enumerate(top_k_experts)]
            
            return selected_experts
        
        def expert_forward(self, x, expert_idx):
            """单个专家的前向传播"""
            expert = self.experts[expert_idx]
            
            # 线性变换: y = x @ W + b
            output = []
            for i in range(self.output_dim):
                value = sum(x[j] * expert['weights'][j][i] for j in range(len(x)))
                value += expert['bias'][i]
                output.append(value)
            
            return output
        
        def forward(self, x):
            """MoE前向传播"""
            # 1. 计算门控权重
            gate_weights = self.gating_network(x)
            
            # 2. 选择Top-K专家
            selected_experts = self.select_top_k_experts(gate_weights)
            
            # 3. 计算选中专家的输出
            expert_outputs = []
            expert_info = []
            
            for expert_idx, weight in selected_experts:
                output = self.expert_forward(x, expert_idx)
                expert_outputs.append(output)
                expert_info.append((expert_idx, weight))
            
            # 4. 加权融合专家输出
            final_output = [0.0] * self.output_dim
            
            for i, (output, (expert_idx, weight)) in enumerate(zip(expert_outputs, expert_info)):
                for j in range(self.output_dim):
                    final_output[j] += weight * output[j]
            
            return final_output, expert_info, gate_weights
        
        def compute_load_balancing_loss(self, gate_weights_batch):
            """计算负载均衡损失"""
            # 计算每个专家的平均门控权重
            avg_gate_weights = [0.0] * self.num_experts
            batch_size = len(gate_weights_batch)
            
            for gate_weights in gate_weights_batch:
                for i in range(self.num_experts):
                    avg_gate_weights[i] += gate_weights[i] / batch_size
            
            # 理想情况下每个专家应该有相等的权重
            target_weight = 1.0 / self.num_experts
            
            # 计算负载不均衡度
            imbalance = sum((w - target_weight) ** 2 for w in avg_gate_weights)
            
            return imbalance, avg_gate_weights
    
    # MoE演示
    print("混合专家模型演示:")
    
    # 创建MoE模型
    moe = MixtureOfExperts(input_dim=10, output_dim=5, num_experts=4, top_k=2)
    
    # 模拟一批输入
    batch_inputs = []
    batch_gate_weights = []
    
    for i in range(3):
        # 生成不同类型的输入
        if i == 0:
            # 类型1：前半部分非零
            x = [random.uniform(0.5, 1.0) for _ in range(5)] + [0.0] * 5
        elif i == 1:
            # 类型2：后半部分非零
            x = [0.0] * 5 + [random.uniform(0.5, 1.0) for _ in range(5)]
        else:
            # 类型3：随机分布
            x = [random.uniform(-0.5, 0.5) for _ in range(10)]
        
        batch_inputs.append(x)
    
    print(f"\n处理{len(batch_inputs)}个输入样本:")
    
    for i, x in enumerate(batch_inputs):
        output, expert_info, gate_weights = moe.forward(x)
        batch_gate_weights.append(gate_weights)
        
        print(f"\n样本{i+1}:")
        print(f"  输入特征: [{', '.join(f'{v:.2f}' for v in x[:5])}...{', '.join(f'{v:.2f}' for v in x[-5:])}]")
        print(f"  门控权重: [{', '.join(f'{w:.3f}' for w in gate_weights)}]")
        print(f"  选中专家: {[(idx, f'{w:.3f}') for idx, w in expert_info]}")
        print(f"  输出: [{', '.join(f'{v:.3f}' for v in output)}]")
    
    # 计算负载均衡
    imbalance, avg_weights = moe.compute_load_balancing_loss(batch_gate_weights)
    
    print(f"\n负载均衡分析:")
    print(f"  每个专家的平均权重: [{', '.join(f'{w:.3f}' for w in avg_weights)}]")
    print(f"  理想权重: {1.0/moe.num_experts:.3f}")
    print(f"  负载不均衡度: {imbalance:.6f}")
    
    print(f"\nMoE的优势:")
    print(f"• 模型容量：增加专家数量而不增加计算")
    print(f"• 专业化：不同专家学习不同模式")  
    print(f"• 效率：稀疏激活，只使用Top-K专家")
    print(f"• 可扩展：容易扩展到大规模模型")

def main():
    """主函数"""
    print("前沿深度学习架构")
    print("=" * 70)
    
    cutting_edge_intro()
    transformer_detailed_implementation()
    vision_transformer_implementation()
    efficient_neural_architecture()
    mixture_of_experts()
    
    print("\n" + "=" * 70)
    print("🎯 前沿架构总结")
    print()
    print("掌握的前沿技术:")
    print("• Transformer：自注意力机制的完整实现")
    print("• Vision Transformer：图像领域的Transformer应用")
    print("• EfficientNet：复合缩放与高效架构设计")
    print("• 混合专家：稀疏激活的大规模模型")
    print()
    print("这些前沿架构代表了深度学习的最新发展方向，")
    print("理解并掌握它们将帮助你跟上技术前沿，")
    print("为研究和应用提供强大的工具！")

if __name__ == "__main__":
    main()