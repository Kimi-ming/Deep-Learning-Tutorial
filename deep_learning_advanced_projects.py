# -*- coding: utf-8 -*-
"""
深度学习高级实践项目 - NAS、VAE、元学习

包含：神经网络架构搜索、变分自编码器、元学习等挑战性项目实现。
"""
# Advanced Deep Learning Projects: 挑战性项目实现

import random
import math
import json

def advanced_projects_intro():
    """高级项目介绍"""
    print("=== 深度学习高级实践项目 ===")
    print("通过挑战性项目深入掌握深度学习技术")
    print()
    print("项目特点:")
    print("- 综合性：整合多种技术")
    print("- 实用性：解决实际问题")
    print("- 挑战性：深度理解原理")
    print("- 前沿性：采用最新技术")
    print()
    print("包含项目:")
    print("- 神经网络架构搜索 (NAS)")
    print("- 变分自编码器 (VAE) 实现")
    print("- 对抗训练与鲁棒性")
    print("- 元学习与少样本学习")
    print("- 神经常微分方程 (Neural ODE)")
    print()

def neural_architecture_search_project():
    """神经网络架构搜索项目"""
    print("\n" + "="*70)
    print("项目 1: 神经网络架构搜索 (NAS)")
    print("="*70)
    
    print("项目目标:")
    print("- 自动搜索最优网络架构")
    print("- 理解架构空间设计")
    print("- 实现演化/强化学习搜索策略")
    print("- 平衡性能与效率")
    print()
    
    class NeuralArchitectureSearch:
        """神经架构搜索实现"""
        
        def __init__(self, search_space_config):
            self.search_space = search_space_config
            self.architecture_history = []
            self.performance_history = []
            self.generation = 0
            
        def define_search_space(self):
            """定义搜索空间"""
            search_space = {
                'num_layers': [3, 4, 5, 6, 7, 8],
                'layer_types': ['conv', 'depthwise_conv', 'mbconv', 'residual'],
                'activation_functions': ['relu', 'swish', 'gelu'],
                'channel_sizes': [16, 32, 64, 128, 256],
                'kernel_sizes': [3, 5, 7],
                'skip_connections': [True, False]
            }
            return search_space
        
        def sample_architecture(self):
            """从搜索空间随机采样架构"""
            space = self.define_search_space()
            
            architecture = {
                'id': f"arch_{len(self.architecture_history):04d}",
                'num_layers': random.choice(space['num_layers']),
                'layers': []
            }
            
            for layer_idx in range(architecture['num_layers']):
                layer_config = {
                    'layer_id': layer_idx,
                    'type': random.choice(space['layer_types']),
                    'activation': random.choice(space['activation_functions']),
                    'channels': random.choice(space['channel_sizes']),
                    'kernel_size': random.choice(space['kernel_sizes']),
                    'skip_connection': random.choice(space['skip_connections'])
                }
                architecture['layers'].append(layer_config)
            
            return architecture
        
        def encode_architecture(self, architecture):
            """将架构编码为固定长度向量"""
            # 为演示，创建简化的编码
            encoding = []
            
            # 编码层数
            encoding.append(architecture['num_layers'] / 10.0)
            
            # 编码每层的配置 (填充到固定长度)
            max_layers = 8
            layer_encoding_size = 6  # 每层6个特征
            
            for i in range(max_layers):
                if i < len(architecture['layers']):
                    layer = architecture['layers'][i]
                    
                    # 层类型 (one-hot编码)
                    layer_types = ['conv', 'depthwise_conv', 'mbconv', 'residual']
                    type_encoding = [1.0 if layer['type'] == t else 0.0 for t in layer_types]
                    
                    # 其他特征
                    other_features = [
                        layer['channels'] / 256.0,  # 归一化通道数
                        layer['kernel_size'] / 7.0,  # 归一化核大小
                        1.0 if layer['skip_connection'] else 0.0
                    ]
                    
                    # 只取前几个特征以简化
                    encoding.extend(type_encoding[:2] + other_features[:4])
                else:
                    # 填充空层
                    encoding.extend([0.0] * layer_encoding_size)
            
            return encoding
        
        def evaluate_architecture(self, architecture):
            """评估架构性能 (模拟)"""
            # 在实际NAS中，这里会训练模型并评估性能
            # 这里我们用启发式方法模拟性能评估
            
            score = 0.0
            complexity_penalty = 0.0
            
            # 基于层数的性能估计
            num_layers = architecture['num_layers']
            if num_layers >= 4 and num_layers <= 6:
                score += 0.3  # 适中深度较好
            elif num_layers > 6:
                complexity_penalty += (num_layers - 6) * 0.05
            
            # 基于层类型的性能估计
            layer_type_scores = {
                'conv': 0.15,
                'depthwise_conv': 0.12,
                'mbconv': 0.18,
                'residual': 0.20
            }
            
            for layer in architecture['layers']:
                score += layer_type_scores.get(layer['type'], 0.1)
                
                # 通道数影响
                if layer['channels'] >= 64 and layer['channels'] <= 128:
                    score += 0.05
                elif layer['channels'] > 256:
                    complexity_penalty += 0.02
                
                # Skip connection奖励
                if layer['skip_connection'] and layer['type'] in ['conv', 'residual']:
                    score += 0.03
            
            # 添加随机噪声模拟训练不确定性
            noise = random.gauss(0, 0.05)
            
            # 最终分数
            final_score = max(0.0, min(1.0, score - complexity_penalty + noise))
            
            # 计算复杂度指标
            complexity = self.calculate_complexity(architecture)
            
            return {
                'accuracy': final_score,
                'complexity': complexity,
                'efficiency': final_score / max(complexity, 0.1)
            }
        
        def calculate_complexity(self, architecture):
            """计算架构复杂度"""
            complexity = 0.0
            
            for layer in architecture['layers']:
                # 基于通道数和核大小估算复杂度
                layer_complexity = (layer['channels'] * layer['kernel_size']**2) / 10000.0
                complexity += layer_complexity
            
            return complexity
        
        def evolutionary_search(self, population_size=20, generations=10, mutation_rate=0.3):
            """演化搜索策略"""
            print(f"开始演化搜索:")
            print(f"  种群大小: {population_size}")
            print(f"  演化代数: {generations}")
            print(f"  变异率: {mutation_rate}")
            
            # 初始化种群
            population = []
            for _ in range(population_size):
                arch = self.sample_architecture()
                performance = self.evaluate_architecture(arch)
                population.append((arch, performance))
            
            best_architectures = []
            
            for gen in range(generations):
                print(f"\n  第 {gen+1} 代:")
                
                # 评估种群
                population.sort(key=lambda x: x[1]['efficiency'], reverse=True)
                
                # 记录最佳架构
                best_arch, best_perf = population[0]
                best_architectures.append((best_arch.copy(), best_perf.copy()))
                
                print(f"    最佳效率: {best_perf['efficiency']:.4f}")
                print(f"    最佳精度: {best_perf['accuracy']:.4f}")
                print(f"    复杂度: {best_perf['complexity']:.4f}")
                
                # 选择策略：保留前50%
                survivors = population[:population_size//2]
                
                # 生成新种群
                new_population = survivors[:]
                
                while len(new_population) < population_size:
                    # 选择父代
                    parent1 = random.choice(survivors)[0]
                    parent2 = random.choice(survivors)[0]
                    
                    # 交叉
                    child = self.crossover(parent1, parent2)
                    
                    # 变异
                    if random.random() < mutation_rate:
                        child = self.mutate(child)
                    
                    # 评估子代
                    child_performance = self.evaluate_architecture(child)
                    new_population.append((child, child_performance))
                
                population = new_population
                self.generation = gen + 1
            
            return best_architectures
        
        def crossover(self, parent1, parent2):
            """架构交叉操作"""
            child = {
                'id': f"arch_cross_{self.generation}_{len(self.architecture_history)}",
                'num_layers': random.choice([parent1['num_layers'], parent2['num_layers']]),
                'layers': []
            }
            
            # 混合层配置
            max_layers = min(len(parent1['layers']), len(parent2['layers']))
            
            for i in range(child['num_layers']):
                if i < max_layers:
                    # 随机选择父代的层配置
                    source_parent = random.choice([parent1, parent2])
                    child['layers'].append(source_parent['layers'][i].copy())
                else:
                    # 如果超出父代层数，随机生成
                    space = self.define_search_space()
                    layer_config = {
                        'layer_id': i,
                        'type': random.choice(space['layer_types']),
                        'activation': random.choice(space['activation_functions']),
                        'channels': random.choice(space['channel_sizes']),
                        'kernel_size': random.choice(space['kernel_sizes']),
                        'skip_connection': random.choice(space['skip_connections'])
                    }
                    child['layers'].append(layer_config)
            
            return child
        
        def mutate(self, architecture):
            """架构变异操作"""
            mutated = json.loads(json.dumps(architecture))  # 深拷贝
            space = self.define_search_space()
            
            # 随机选择变异类型
            mutation_type = random.choice(['layer_type', 'channels', 'kernel_size', 'skip_connection'])
            
            if mutated['layers']:
                # 随机选择要变异的层
                layer_idx = random.randint(0, len(mutated['layers']) - 1)
                layer = mutated['layers'][layer_idx]
                
                if mutation_type == 'layer_type':
                    layer['type'] = random.choice(space['layer_types'])
                elif mutation_type == 'channels':
                    layer['channels'] = random.choice(space['channel_sizes'])
                elif mutation_type == 'kernel_size':
                    layer['kernel_size'] = random.choice(space['kernel_sizes'])
                elif mutation_type == 'skip_connection':
                    layer['skip_connection'] = random.choice(space['skip_connections'])
            
            return mutated
    
    # NAS项目演示
    print("神经架构搜索项目演示:")
    
    # 创建NAS实例
    nas = NeuralArchitectureSearch({})
    
    # 演示架构采样和评估
    print(f"\n1. 架构采样和评估:")
    for i in range(3):
        arch = nas.sample_architecture()
        performance = nas.evaluate_architecture(arch)
        
        print(f"\n架构 {i+1} ({arch['id']}):")
        print(f"  层数: {arch['num_layers']}")
        print(f"  层配置示例: {arch['layers'][0] if arch['layers'] else 'None'}")
        print(f"  性能: 精度={performance['accuracy']:.4f}, 复杂度={performance['complexity']:.4f}, 效率={performance['efficiency']:.4f}")
    
    # 运行演化搜索 (小规模演示)
    print(f"\n2. 演化搜索:")
    best_architectures = nas.evolutionary_search(population_size=8, generations=3, mutation_rate=0.3)
    
    print(f"\n3. 搜索结果分析:")
    print(f"找到 {len(best_architectures)} 代最佳架构:")
    
    for i, (arch, perf) in enumerate(best_architectures):
        print(f"  第{i+1}代最佳: 层数={arch['num_layers']}, 效率={perf['efficiency']:.4f}")
    
    # 显示最终最佳架构
    if best_architectures:
        final_best = max(best_architectures, key=lambda x: x[1]['efficiency'])
        print(f"\n4. 最终最佳架构:")
        print(f"  ID: {final_best[0]['id']}")
        print(f"  层数: {final_best[0]['num_layers']}")
        print(f"  效率: {final_best[1]['efficiency']:.4f}")
        print(f"  精度: {final_best[1]['accuracy']:.4f}")
        print(f"  复杂度: {final_best[1]['complexity']:.4f}")

def variational_autoencoder_project():
    """变分自编码器项目"""
    print("\n" + "="*70)
    print("项目 2: 变分自编码器 (VAE)")
    print("="*70)
    
    print("项目目标:")
    print("- 实现完整的VAE架构")
    print("- 理解变分推断原理")
    print("- 生成新的数据样本")
    print("- 学习潜在空间表示")
    print()
    
    class VariationalAutoencoder:
        """变分自编码器实现"""
        
        def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            
            # 初始化编码器网络
            self.encoder_weights = {
                'W1': self._init_weights(input_dim, hidden_dim),
                'b1': [0.0] * hidden_dim,
                'W_mu': self._init_weights(hidden_dim, latent_dim),
                'b_mu': [0.0] * latent_dim,
                'W_logvar': self._init_weights(hidden_dim, latent_dim),
                'b_logvar': [0.0] * latent_dim
            }
            
            # 初始化解码器网络
            self.decoder_weights = {
                'W1': self._init_weights(latent_dim, hidden_dim),
                'b1': [0.0] * hidden_dim,
                'W2': self._init_weights(hidden_dim, input_dim),
                'b2': [0.0] * input_dim
            }
            
            print(f"VAE 初始化:")
            print(f"  输入维度: {input_dim}")
            print(f"  潜在维度: {latent_dim}")
            print(f"  隐藏维度: {hidden_dim}")
            
        def _init_weights(self, in_dim, out_dim):
            """Xavier初始化"""
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            return [[random.uniform(-limit, limit) for _ in range(out_dim)] 
                   for _ in range(in_dim)]
        
        def relu(self, x):
            """ReLU激活函数"""
            return [max(0, xi) for xi in x]
        
        def sigmoid(self, x):
            """Sigmoid激活函数"""
            return [1.0 / (1.0 + math.exp(-max(-500, min(500, xi)))) for xi in x]
        
        def linear_layer(self, x, weights, bias):
            """线性层"""
            output = []
            for i in range(len(bias)):
                value = sum(x[j] * weights[j][i] for j in range(len(x))) + bias[i]
                output.append(value)
            return output
        
        def encoder(self, x):
            """编码器：x -> (μ, log_σ²)"""
            # 隐藏层
            h1 = self.linear_layer(x, self.encoder_weights['W1'], self.encoder_weights['b1'])
            h1_activated = self.relu(h1)
            
            # 均值分支
            mu = self.linear_layer(h1_activated, self.encoder_weights['W_mu'], self.encoder_weights['b_mu'])
            
            # 对数方差分支
            log_var = self.linear_layer(h1_activated, self.encoder_weights['W_logvar'], self.encoder_weights['b_logvar'])
            
            return mu, log_var
        
        def reparameterization_trick(self, mu, log_var):
            """重参数化技巧：z = μ + σ ⊙ ε"""
            std = [math.exp(0.5 * lv) for lv in log_var]
            epsilon = [random.gauss(0, 1) for _ in range(len(mu))]
            
            z = [mu[i] + std[i] * epsilon[i] for i in range(len(mu))]
            
            return z, epsilon
        
        def decoder(self, z):
            """解码器：z -> x̂"""
            # 隐藏层
            h1 = self.linear_layer(z, self.decoder_weights['W1'], self.decoder_weights['b1'])
            h1_activated = self.relu(h1)
            
            # 输出层
            output = self.linear_layer(h1_activated, self.decoder_weights['W2'], self.decoder_weights['b2'])
            output_activated = self.sigmoid(output)  # 假设输入在[0,1]范围
            
            return output_activated
        
        def forward(self, x):
            """VAE前向传播"""
            # 编码
            mu, log_var = self.encoder(x)
            
            # 重参数化
            z, epsilon = self.reparameterization_trick(mu, log_var)
            
            # 解码
            x_reconstructed = self.decoder(z)
            
            return x_reconstructed, mu, log_var, z
        
        def compute_loss(self, x, x_reconstructed, mu, log_var):
            """计算VAE损失"""
            # 重构损失 (二元交叉熵)
            reconstruction_loss = 0.0
            for i in range(len(x)):
                x_i = max(1e-8, min(1-1e-8, x[i]))  # 数值稳定
                x_recon_i = max(1e-8, min(1-1e-8, x_reconstructed[i]))
                
                reconstruction_loss += -(x_i * math.log(x_recon_i) + 
                                       (1 - x_i) * math.log(1 - x_recon_i))
            
            # KL散度损失
            kl_loss = 0.0
            for i in range(len(mu)):
                kl_loss += 0.5 * (math.exp(log_var[i]) + mu[i]**2 - 1 - log_var[i])
            
            # 总损失
            total_loss = reconstruction_loss + kl_loss
            
            return {
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss
            }
        
        def generate_samples(self, num_samples=5):
            """从先验分布生成新样本"""
            generated_samples = []
            
            for _ in range(num_samples):
                # 从标准正态分布采样
                z = [random.gauss(0, 1) for _ in range(self.latent_dim)]
                
                # 解码生成样本
                x_generated = self.decoder(z)
                generated_samples.append(x_generated)
            
            return generated_samples
        
        def interpolate_in_latent_space(self, x1, x2, num_steps=5):
            """在潜在空间中插值"""
            # 编码两个输入
            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)
            
            # 在潜在空间中插值
            interpolated_samples = []
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interpolated = [mu1[j] * (1 - alpha) + mu2[j] * alpha 
                                for j in range(len(mu1))]
                
                # 解码插值点
                x_interpolated = self.decoder(z_interpolated)
                interpolated_samples.append(x_interpolated)
            
            return interpolated_samples
    
    # VAE项目演示
    print("变分自编码器项目演示:")
    
    # 创建VAE模型
    vae = VariationalAutoencoder(input_dim=20, latent_dim=5, hidden_dim=15)  # 小尺寸用于演示
    
    # 生成模拟数据
    def generate_synthetic_data(num_samples=10, dim=20):
        """生成合成数据"""
        data = []
        for _ in range(num_samples):
            # 生成具有某种结构的数据
            sample = []
            pattern_type = random.choice(['linear', 'quadratic', 'sine'])
            
            for i in range(dim):
                if pattern_type == 'linear':
                    value = i / dim + random.gauss(0, 0.1)
                elif pattern_type == 'quadratic':
                    value = (i / dim) ** 2 + random.gauss(0, 0.1)
                else:  # sine
                    value = 0.5 * (1 + math.sin(2 * math.pi * i / dim)) + random.gauss(0, 0.1)
                
                # 将值限制在[0,1]范围
                value = max(0, min(1, value))
                sample.append(value)
            
            data.append(sample)
        
        return data
    
    # 生成训练数据
    train_data = generate_synthetic_data(num_samples=6, dim=vae.input_dim)
    
    print(f"\n1. 数据处理和前向传播:")
    print(f"训练数据维度: {len(train_data)} × {len(train_data[0])}")
    
    # 处理训练样本
    losses = []
    
    for i, x in enumerate(train_data[:3]):  # 只处理前3个样本做演示
        # 前向传播
        x_recon, mu, log_var, z = vae.forward(x)
        
        # 计算损失
        loss_info = vae.compute_loss(x, x_recon, mu, log_var)
        losses.append(loss_info)
        
        print(f"\n样本 {i+1}:")
        print(f"  原始输入: [{', '.join(f'{v:.3f}' for v in x[:5])}...]")
        print(f"  重构输出: [{', '.join(f'{v:.3f}' for v in x_recon[:5])}...]")
        print(f"  潜在编码: [{', '.join(f'{v:.3f}' for v in z)}]")
        print(f"  总损失: {loss_info['total_loss']:.4f}")
        print(f"  重构损失: {loss_info['reconstruction_loss']:.4f}")
        print(f"  KL损失: {loss_info['kl_loss']:.4f}")
    
    # 生成新样本
    print(f"\n2. 生成新样本:")
    generated_samples = vae.generate_samples(num_samples=3)
    
    for i, sample in enumerate(generated_samples):
        print(f"  生成样本 {i+1}: [{', '.join(f'{v:.3f}' for v in sample[:5])}...]")
    
    # 潜在空间插值
    if len(train_data) >= 2:
        print(f"\n3. 潜在空间插值:")
        interpolated = vae.interpolate_in_latent_space(train_data[0], train_data[1], num_steps=3)
        
        print(f"  起始样本: [{', '.join(f'{v:.3f}' for v in train_data[0][:5])}...]")
        for i, sample in enumerate(interpolated):
            print(f"  插值步骤 {i+1}: [{', '.join(f'{v:.3f}' for v in sample[:5])}...]")
        print(f"  结束样本: [{', '.join(f'{v:.3f}' for v in train_data[1][:5])}...]")
    
    print(f"\n4. VAE分析:")
    avg_total_loss = sum(l['total_loss'] for l in losses) / len(losses)
    avg_recon_loss = sum(l['reconstruction_loss'] for l in losses) / len(losses)
    avg_kl_loss = sum(l['kl_loss'] for l in losses) / len(losses)
    
    print(f"  平均总损失: {avg_total_loss:.4f}")
    print(f"  平均重构损失: {avg_recon_loss:.4f}")
    print(f"  平均KL损失: {avg_kl_loss:.4f}")
    print(f"  KL/重构比率: {avg_kl_loss/avg_recon_loss:.4f}")

def meta_learning_project():
    """元学习项目"""
    print("\n" + "="*70)
    print("项目 3: 元学习与少样本学习")
    print("="*70)
    
    print("项目目标:")
    print("- 实现Model-Agnostic Meta-Learning (MAML)")
    print("- 理解学习如何学习的概念")
    print("- 快速适应新任务")
    print("- 少样本学习能力")
    print()
    
    class MAML:
        """Model-Agnostic Meta-Learning实现"""
        
        def __init__(self, input_dim=5, hidden_dim=10, output_dim=1, 
                     meta_lr=0.001, inner_lr=0.01):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.meta_lr = meta_lr
            self.inner_lr = inner_lr
            
            # 初始化元参数
            self.meta_params = {
                'W1': self._init_weights(input_dim, hidden_dim),
                'b1': [0.0] * hidden_dim,
                'W2': self._init_weights(hidden_dim, output_dim),
                'b2': [0.0] * output_dim
            }
            
            print(f"MAML 初始化:")
            print(f"  网络结构: {input_dim} -> {hidden_dim} -> {output_dim}")
            print(f"  元学习率: {meta_lr}")
            print(f"  内层学习率: {inner_lr}")
            
        def _init_weights(self, in_dim, out_dim):
            """Xavier初始化"""
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            return [[random.uniform(-limit, limit) for _ in range(out_dim)] 
                   for _ in range(in_dim)]
        
        def forward(self, x, params):
            """前向传播"""
            # 隐藏层
            h1 = []
            for i in range(len(params['b1'])):
                value = sum(x[j] * params['W1'][j][i] for j in range(len(x))) + params['b1'][i]
                h1.append(max(0, value))  # ReLU
            
            # 输出层
            output = []
            for i in range(len(params['b2'])):
                value = sum(h1[j] * params['W2'][j][i] for j in range(len(h1))) + params['b2'][i]
                output.append(value)
            
            return output
        
        def compute_loss(self, predictions, targets):
            """计算均方误差损失"""
            loss = 0.0
            for pred, target in zip(predictions, targets):
                if isinstance(pred, list):
                    pred = pred[0]  # 假设单输出
                loss += (pred - target) ** 2
            return loss / len(predictions)
        
        def compute_gradients(self, x_batch, y_batch, params):
            """计算梯度 (简化的数值梯度)"""
            gradients = {}
            eps = 1e-5
            
            # 计算原始损失
            predictions = [self.forward(x, params) for x in x_batch]
            original_loss = self.compute_loss(predictions, y_batch)
            
            # 对每个参数计算数值梯度
            for param_name in params:
                if isinstance(params[param_name], list):
                    if isinstance(params[param_name][0], list):
                        # 二维参数 (权重矩阵)
                        gradients[param_name] = []
                        for i in range(len(params[param_name])):
                            grad_row = []
                            for j in range(len(params[param_name][i])):
                                # 向前扰动
                                params[param_name][i][j] += eps
                                pred_plus = [self.forward(x, params) for x in x_batch]
                                loss_plus = self.compute_loss(pred_plus, y_batch)
                                
                                # 向后扰动
                                params[param_name][i][j] -= 2 * eps
                                pred_minus = [self.forward(x, params) for x in x_batch]
                                loss_minus = self.compute_loss(pred_minus, y_batch)
                                
                                # 恢复原值
                                params[param_name][i][j] += eps
                                
                                # 计算梯度
                                grad = (loss_plus - loss_minus) / (2 * eps)
                                grad_row.append(grad)
                            gradients[param_name].append(grad_row)
                    else:
                        # 一维参数 (偏置)
                        gradients[param_name] = []
                        for i in range(len(params[param_name])):
                            # 向前扰动
                            params[param_name][i] += eps
                            pred_plus = [self.forward(x, params) for x in x_batch]
                            loss_plus = self.compute_loss(pred_plus, y_batch)
                            
                            # 向后扰动
                            params[param_name][i] -= 2 * eps
                            pred_minus = [self.forward(x, params) for x in x_batch]
                            loss_minus = self.compute_loss(pred_minus, y_batch)
                            
                            # 恢复原值
                            params[param_name][i] += eps
                            
                            # 计算梯度
                            grad = (loss_plus - loss_minus) / (2 * eps)
                            gradients[param_name].append(grad)
            
            return gradients
        
        def update_parameters(self, params, gradients, learning_rate):
            """更新参数"""
            updated_params = {}
            
            for param_name in params:
                if isinstance(params[param_name][0], list):
                    # 二维参数
                    updated_params[param_name] = []
                    for i in range(len(params[param_name])):
                        updated_row = []
                        for j in range(len(params[param_name][i])):
                            new_val = params[param_name][i][j] - learning_rate * gradients[param_name][i][j]
                            updated_row.append(new_val)
                        updated_params[param_name].append(updated_row)
                else:
                    # 一维参数
                    updated_params[param_name] = []
                    for i in range(len(params[param_name])):
                        new_val = params[param_name][i] - learning_rate * gradients[param_name][i]
                        updated_params[param_name].append(new_val)
            
            return updated_params
        
        def inner_loop_adaptation(self, support_x, support_y, num_steps=1):
            """内层循环：快速适应"""
            adapted_params = {}
            
            # 深拷贝元参数
            for param_name in self.meta_params:
                if isinstance(self.meta_params[param_name][0], list):
                    adapted_params[param_name] = [row[:] for row in self.meta_params[param_name]]
                else:
                    adapted_params[param_name] = self.meta_params[param_name][:]
            
            # 执行几步梯度下降
            for step in range(num_steps):
                gradients = self.compute_gradients(support_x, support_y, adapted_params)
                adapted_params = self.update_parameters(adapted_params, gradients, self.inner_lr)
            
            return adapted_params
        
        def meta_update(self, task_batch):
            """元更新：更新元参数"""
            meta_gradients = {}
            
            # 初始化元梯度
            for param_name in self.meta_params:
                if isinstance(self.meta_params[param_name][0], list):
                    meta_gradients[param_name] = [[0.0 for _ in row] for row in self.meta_params[param_name]]
                else:
                    meta_gradients[param_name] = [0.0] * len(self.meta_params[param_name])
            
            # 对每个任务计算元梯度
            for task in task_batch:
                support_x, support_y, query_x, query_y = task
                
                # 内层适应
                adapted_params = self.inner_loop_adaptation(support_x, support_y)
                
                # 在查询集上计算梯度
                query_gradients = self.compute_gradients(query_x, query_y, adapted_params)
                
                # 累积元梯度
                for param_name in meta_gradients:
                    if isinstance(meta_gradients[param_name][0], list):
                        for i in range(len(meta_gradients[param_name])):
                            for j in range(len(meta_gradients[param_name][i])):
                                meta_gradients[param_name][i][j] += query_gradients[param_name][i][j]
                    else:
                        for i in range(len(meta_gradients[param_name])):
                            meta_gradients[param_name][i] += query_gradients[param_name][i]
            
            # 平均元梯度
            num_tasks = len(task_batch)
            for param_name in meta_gradients:
                if isinstance(meta_gradients[param_name][0], list):
                    for i in range(len(meta_gradients[param_name])):
                        for j in range(len(meta_gradients[param_name][i])):
                            meta_gradients[param_name][i][j] /= num_tasks
                else:
                    for i in range(len(meta_gradients[param_name])):
                        meta_gradients[param_name][i] /= num_tasks
            
            # 更新元参数
            self.meta_params = self.update_parameters(self.meta_params, meta_gradients, self.meta_lr)
    
    # 生成少样本学习任务
    def generate_regression_task(task_type='linear', num_support=3, num_query=2):
        """生成回归任务"""
        x_range = [-2, 2]
        
        if task_type == 'linear':
            # 线性函数 y = ax + b
            a = random.uniform(-2, 2)
            b = random.uniform(-1, 1)
            func = lambda x: a * x + b
        elif task_type == 'quadratic':
            # 二次函数 y = ax² + bx + c
            a = random.uniform(-0.5, 0.5)
            b = random.uniform(-1, 1)
            c = random.uniform(-1, 1)
            func = lambda x: a * x**2 + b * x + c
        else:  # sine
            # 正弦函数 y = a * sin(bx + c)
            a = random.uniform(0.5, 2)
            b = random.uniform(0.5, 3)
            c = random.uniform(0, 2*math.pi)
            func = lambda x: a * math.sin(b * x + c)
        
        # 生成支持集
        support_x = []
        support_y = []
        for _ in range(num_support):
            x = random.uniform(x_range[0], x_range[1])
            y = func(x) + random.gauss(0, 0.1)  # 添加噪声
            support_x.append([x])  # 包装为列表
            support_y.append(y)
        
        # 生成查询集
        query_x = []
        query_y = []
        for _ in range(num_query):
            x = random.uniform(x_range[0], x_range[1])
            y = func(x) + random.gauss(0, 0.1)
            query_x.append([x])
            query_y.append(y)
        
        return support_x, support_y, query_x, query_y
    
    # MAML项目演示
    print("元学习项目演示:")
    
    # 创建MAML模型
    maml = MAML(input_dim=1, hidden_dim=8, output_dim=1)
    
    # 生成任务批次
    print(f"\n1. 生成多样化任务:")
    task_types = ['linear', 'quadratic', 'sine']
    task_batch = []
    
    for i, task_type in enumerate(task_types):
        task = generate_regression_task(task_type, num_support=3, num_query=2)
        task_batch.append(task)
        
        support_x, support_y, query_x, query_y = task
        print(f"  任务{i+1} ({task_type}):")
        print(f"    支持集: X={[x[0] for x in support_x]}, Y={[f'{y:.2f}' for y in support_y]}")
        print(f"    查询集: X={[x[0] for x in query_x]}, Y={[f'{y:.2f}' for y in query_y]}")
    
    # 元训练前的性能
    print(f"\n2. 元训练前性能评估:")
    total_error_before = 0
    
    for i, task in enumerate(task_batch):
        support_x, support_y, query_x, query_y = task
        
        # 使用原始元参数预测
        predictions = [maml.forward(x, maml.meta_params) for x in query_x]
        error = maml.compute_loss(predictions, query_y)
        total_error_before += error
        
        print(f"  任务{i+1}误差: {error:.4f}")
    
    avg_error_before = total_error_before / len(task_batch)
    print(f"  平均误差: {avg_error_before:.4f}")
    
    # 执行元更新 (简化演示)
    print(f"\n3. 执行元更新...")
    maml.meta_update(task_batch)
    print("  元参数已更新")
    
    # 元训练后的性能
    print(f"\n4. 元训练后性能评估:")
    total_error_after = 0
    
    for i, task in enumerate(task_batch):
        support_x, support_y, query_x, query_y = task
        
        # 快速适应
        adapted_params = maml.inner_loop_adaptation(support_x, support_y, num_steps=1)
        
        # 在查询集上评估
        predictions = [maml.forward(x, adapted_params) for x in query_x]
        error = maml.compute_loss(predictions, query_y)
        total_error_after += error
        
        print(f"  任务{i+1}误差: {error:.4f} (适应后)")
    
    avg_error_after = total_error_after / len(task_batch)
    print(f"  平均误差: {avg_error_after:.4f}")
    
    # 性能改进分析
    improvement = (avg_error_before - avg_error_after) / avg_error_before * 100
    print(f"\n5. 元学习效果:")
    print(f"  误差改进: {improvement:.1f}%")
    print(f"  快速适应能力: {'显著' if improvement > 10 else '一般' if improvement > 0 else '需要更多训练'}")

def main():
    """主函数"""
    print("深度学习高级实践项目")
    print("=" * 70)
    
    advanced_projects_intro()
    neural_architecture_search_project()
    variational_autoencoder_project()
    meta_learning_project()
    
    print("\n" + "=" * 70)
    print("高级项目总结")
    print()
    print("完成的挑战性项目:")
    print("- 神经架构搜索：自动化网络设计")
    print("- 变分自编码器：生成模型与表示学习")
    print("- 元学习：学习如何快速学习")
    print()
    print("获得的核心技能:")
    print("- 复杂系统设计与实现能力")
    print("- 前沿技术的深度理解")
    print("- 问题分解与工程实践能力")
    print("- 理论与实践结合的综合能力")
    print()
    print("这些高级项目展示了深度学习的强大潜力，")
    print("通过实际动手实现，你已经掌握了")
    print("最前沿的深度学习技术！")

if __name__ == "__main__":
    main()