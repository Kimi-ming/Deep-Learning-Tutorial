# -*- coding: utf-8 -*-
# 深度学习数学原理深度解析
# Mathematical Foundations of Deep Learning: 深入理解数学基础

import random
import math
import numpy as np

def mathematical_foundations_intro():
    """数学基础介绍"""
    print("=== 深度学习数学原理深度解析 ===")
    print("深入理解深度学习背后的数学原理")
    print()
    print("核心数学概念:")
    print("• 多元微积分与链式法则")
    print("• 线性代数与矩阵运算")
    print("• 概率论与信息论")
    print("• 优化理论与凸优化")
    print("• 泛函分析与变分法")
    print()

def chain_rule_detailed_analysis():
    """链式法则详细分析"""
    print("\n" + "="*60)
    print("链式法则与反向传播的数学原理")
    print("="*60)
    
    print("设神经网络的复合函数为: y = f(g(h(x)))")
    print("其中每一层都是前一层的函数")
    print()
    
    print("1. 单变量链式法则:")
    print("   dy/dx = (dy/dg) × (dg/dh) × (dh/dx)")
    print()
    
    print("2. 多变量链式法则:")
    print("   对于 z = f(x,y), x = g(t), y = h(t)")
    print("   dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)")
    print()
    
    class ChainRuleDemo:
        """链式法则演示"""
        
        def __init__(self):
            self.computation_graph = {}
            self.gradients = {}
            
        def forward_pass(self, x):
            """前向传播演示"""
            # 构建计算图: y = sin(x^2 + 3x)
            a = x * x  # a = x^2
            b = 3 * x  # b = 3x  
            c = a + b  # c = x^2 + 3x
            y = math.sin(c)  # y = sin(c)
            
            # 保存中间结果用于反向传播
            self.computation_graph = {
                'x': x, 'a': a, 'b': b, 'c': c, 'y': y
            }
            
            return y
            
        def backward_pass(self):
            """反向传播演示"""
            x = self.computation_graph['x']
            c = self.computation_graph['c']
            
            # 反向计算梯度
            # dy/dy = 1
            dy_dy = 1
            
            # dy/dc = cos(c)  
            dy_dc = math.cos(c)
            
            # dc/da = 1, dc/db = 1
            dc_da = 1
            dc_db = 1
            
            # da/dx = 2x, db/dx = 3
            da_dx = 2 * x
            db_dx = 3
            
            # 应用链式法则: dy/dx = (dy/dc) × [(dc/da)(da/dx) + (dc/db)(db/dx)]
            dy_dx = dy_dc * (dc_da * da_dx + dc_db * db_dx)
            
            self.gradients = {
                'dy/dc': dy_dc,
                'dc/da': dc_da,
                'dc/db': dc_db, 
                'da/dx': da_dx,
                'db/dx': db_dx,
                'dy/dx': dy_dx
            }
            
            return dy_dx
            
        def analytical_gradient(self, x):
            """解析梯度计算（用于验证）"""
            # y = sin(x^2 + 3x)
            # dy/dx = cos(x^2 + 3x) × (2x + 3)
            return math.cos(x*x + 3*x) * (2*x + 3)
    
    # 演示链式法则计算
    demo = ChainRuleDemo()
    x_val = 2.0
    
    print(f"演示计算: y = sin(x² + 3x), x = {x_val}")
    
    # 前向传播
    y_val = demo.forward_pass(x_val)
    print(f"前向传播结果: y = {y_val:.6f}")
    
    # 反向传播
    grad_numerical = demo.backward_pass()
    grad_analytical = demo.analytical_gradient(x_val)
    
    print(f"\n梯度计算过程:")
    for name, value in demo.gradients.items():
        print(f"  {name} = {value:.6f}")
    
    print(f"\n梯度验证:")
    print(f"  数值计算: dy/dx = {grad_numerical:.6f}")
    print(f"  解析计算: dy/dx = {grad_analytical:.6f}")
    print(f"  误差: {abs(grad_numerical - grad_analytical):.10f}")

def information_theory_in_deep_learning():
    """信息论在深度学习中的应用"""
    print("\n" + "="*60)
    print("信息论与深度学习")
    print("="*60)
    
    print("核心概念:")
    print("• 熵(Entropy): 信息的度量")
    print("• 交叉熵(Cross-Entropy): 损失函数的理论基础")
    print("• KL散度: 分布之间的距离")
    print("• 互信息: 变量间的依赖关系")
    print()
    
    class InformationTheory:
        """信息论计算工具"""
        
        @staticmethod
        def entropy(probabilities):
            """计算熵 H(X) = -Σ p(x) log p(x)"""
            entropy = 0
            for p in probabilities:
                if p > 0:  # 避免log(0)
                    entropy -= p * math.log2(p)
            return entropy
            
        @staticmethod
        def cross_entropy(true_dist, pred_dist):
            """计算交叉熵 H(P,Q) = -Σ p(x) log q(x)"""
            cross_ent = 0
            for p, q in zip(true_dist, pred_dist):
                if p > 0 and q > 0:
                    cross_ent -= p * math.log2(q)
            return cross_ent
            
        @staticmethod
        def kl_divergence(true_dist, pred_dist):
            """计算KL散度 D_KL(P||Q) = Σ p(x) log(p(x)/q(x))"""
            kl_div = 0
            for p, q in zip(true_dist, pred_dist):
                if p > 0 and q > 0:
                    kl_div += p * math.log2(p / q)
            return kl_div
            
        @staticmethod
        def mutual_information(joint_prob, marginal_x, marginal_y):
            """计算互信息 I(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))"""
            mi = 0
            for i in range(len(joint_prob)):
                for j in range(len(joint_prob[0])):
                    p_xy = joint_prob[i][j]
                    p_x = marginal_x[i]
                    p_y = marginal_y[j]
                    if p_xy > 0 and p_x > 0 and p_y > 0:
                        mi += p_xy * math.log2(p_xy / (p_x * p_y))
            return mi
    
    # 信息论应用演示
    print("信息论计算示例:")
    
    # 1. 熵的计算
    uniform_dist = [0.25, 0.25, 0.25, 0.25]  # 均匀分布
    skewed_dist = [0.7, 0.2, 0.05, 0.05]     # 偏斜分布
    
    entropy_uniform = InformationTheory.entropy(uniform_dist)
    entropy_skewed = InformationTheory.entropy(skewed_dist)
    
    print(f"\n1. 熵计算:")
    print(f"   均匀分布 {uniform_dist}: H = {entropy_uniform:.4f} bits")
    print(f"   偏斜分布 {skewed_dist}: H = {entropy_skewed:.4f} bits")
    print(f"   结论: 均匀分布具有最大熵")
    
    # 2. 交叉熵与KL散度
    true_dist = [0.6, 0.3, 0.1]
    pred_dist = [0.5, 0.4, 0.1]
    
    cross_ent = InformationTheory.cross_entropy(true_dist, pred_dist)
    entropy_true = InformationTheory.entropy(true_dist)
    kl_div = InformationTheory.kl_divergence(true_dist, pred_dist)
    
    print(f"\n2. 交叉熵与KL散度:")
    print(f"   真实分布: {true_dist}")
    print(f"   预测分布: {pred_dist}")
    print(f"   H(P): {entropy_true:.4f}")
    print(f"   H(P,Q): {cross_ent:.4f}")
    print(f"   D_KL(P||Q): {kl_div:.4f}")
    print(f"   验证: H(P,Q) = H(P) + D_KL(P||Q) = {entropy_true + kl_div:.4f}")
    
    print(f"\n在深度学习中的应用:")
    print(f"• 交叉熵损失函数衡量预测分布与真实分布的差异")
    print(f"• KL散度用于变分推断和正则化")
    print(f"• 互信息用于特征选择和表示学习")

def optimization_theory_advanced():
    """优化理论高级内容"""
    print("\n" + "="*60)
    print("优化理论与深度学习")
    print("="*60)
    
    print("高级优化概念:")
    print("• 凸优化与非凸优化")
    print("• 鞍点问题与逃逸策略")
    print("• 二阶优化方法")
    print("• 自适应学习率方法")
    print()
    
    class AdvancedOptimizers:
        """高级优化器实现"""
        
        def __init__(self, params_shape):
            self.params_shape = params_shape
            self.reset()
            
        def reset(self):
            """重置优化器状态"""
            self.m = [0.0] * self.params_shape  # 一阶动量
            self.v = [0.0] * self.params_shape  # 二阶动量
            self.t = 0  # 时间步
            
        def adam_optimizer(self, gradients, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            """Adam优化器详细实现"""
            self.t += 1
            
            updated_params = []
            
            for i in range(len(gradients)):
                # 更新偏置修正的一阶和二阶动量估计
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (gradients[i] ** 2)
                
                # 偏置修正
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)
                
                # 参数更新
                update = learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
                updated_params.append(update)
            
            return updated_params
            
        def rmsprop_optimizer(self, gradients, learning_rate=0.001, beta=0.9, epsilon=1e-8):
            """RMSprop优化器实现"""
            updated_params = []
            
            for i in range(len(gradients)):
                # 更新二阶动量
                self.v[i] = beta * self.v[i] + (1 - beta) * (gradients[i] ** 2)
                
                # 参数更新
                update = learning_rate * gradients[i] / (math.sqrt(self.v[i]) + epsilon)
                updated_params.append(update)
                
            return updated_params
            
        def adagrad_optimizer(self, gradients, learning_rate=0.01, epsilon=1e-8):
            """Adagrad优化器实现"""
            updated_params = []
            
            for i in range(len(gradients)):
                # 累积平方梯度
                self.v[i] += gradients[i] ** 2
                
                # 参数更新
                update = learning_rate * gradients[i] / (math.sqrt(self.v[i]) + epsilon)
                updated_params.append(update)
                
            return updated_params
    
    # 优化器性能比较
    print("优化器性能比较:")
    
    def rosenbrock_function(x, y):
        """Rosenbrock函数: f(x,y) = (a-x)² + b(y-x²)²"""
        a, b = 1, 100
        return (a - x)**2 + b * (y - x**2)**2
    
    def rosenbrock_gradient(x, y):
        """Rosenbrock函数的梯度"""
        a, b = 1, 100
        dx = -2*(a - x) - 4*b*x*(y - x**2)
        dy = 2*b*(y - x**2)
        return [dx, dy]
    
    # 测试不同优化器
    optimizers = {
        'Adam': lambda opt, grad: opt.adam_optimizer(grad, learning_rate=0.01),
        'RMSprop': lambda opt, grad: opt.rmsprop_optimizer(grad, learning_rate=0.01),
        'Adagrad': lambda opt, grad: opt.adagrad_optimizer(grad, learning_rate=0.1)
    }
    
    initial_point = [-1.0, 1.0]
    target_point = [1.0, 1.0]  # 全局最优点
    
    print(f"测试函数: Rosenbrock函数")
    print(f"起始点: {initial_point}")
    print(f"目标点: {target_point}")
    print(f"迭代次数: 100")
    print()
    
    for name, optimizer_func in optimizers.items():
        opt = AdvancedOptimizers(2)
        x, y = initial_point[:]
        
        for iteration in range(100):
            grad = rosenbrock_gradient(x, y)
            updates = optimizer_func(opt, grad)
            x -= updates[0]
            y -= updates[1]
        
        final_value = rosenbrock_function(x, y)
        distance_to_optimum = math.sqrt((x - 1)**2 + (y - 1)**2)
        
        print(f"{name:>8}: 最终点({x:.4f}, {y:.4f}), 函数值={final_value:.6f}, 距离最优={distance_to_optimum:.6f}")

def variational_inference_theory():
    """变分推断理论"""
    print("\n" + "="*60)
    print("变分推断与深度生成模型")
    print("="*60)
    
    print("变分推断核心思想:")
    print("• 用简单分布近似复杂的后验分布")
    print("• 最小化KL散度找到最佳近似")
    print("• ELBO(Evidence Lower Bound)优化")
    print("• 重参数化技巧实现梯度传播")
    print()
    
    class VariationalInference:
        """变分推断实现"""
        
        def __init__(self, latent_dim=2):
            self.latent_dim = latent_dim
            
        def gaussian_kl_divergence(self, mu1, sigma1, mu2=0, sigma2=1):
            """计算两个高斯分布的KL散度"""
            # KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))
            kl = math.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5
            return kl
            
        def elbo_calculation(self, data, mu_encoder, sigma_encoder, reconstruction_loss):
            """计算ELBO (Evidence Lower BOund)"""
            # ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
            
            # 重构损失项 (负对数似然)
            reconstruction_term = -reconstruction_loss
            
            # KL散度项 (正则化项)
            kl_term = 0
            for i in range(len(mu_encoder)):
                kl_term += self.gaussian_kl_divergence(mu_encoder[i], sigma_encoder[i])
            
            elbo = reconstruction_term - kl_term
            return elbo, reconstruction_term, kl_term
            
        def reparameterization_trick(self, mu, sigma):
            """重参数化技巧"""
            # z = μ + σ * ε, where ε ~ N(0,1)
            epsilon = random.gauss(0, 1)
            z = mu + sigma * epsilon
            return z, epsilon
    
    # 变分推断演示
    print("变分自编码器(VAE)原理演示:")
    
    vi = VariationalInference()
    
    # 模拟编码器输出
    mu_encoder = [0.5, -0.3]  # 均值
    sigma_encoder = [0.8, 1.2]  # 标准差
    reconstruction_loss = 2.5
    
    print(f"编码器输出:")
    print(f"  μ = {mu_encoder}")
    print(f"  σ = {sigma_encoder}")
    print(f"重构损失 = {reconstruction_loss}")
    
    # 计算ELBO
    elbo, recon_term, kl_term = vi.elbo_calculation(
        None, mu_encoder, sigma_encoder, reconstruction_loss
    )
    
    print(f"\nELBO计算:")
    print(f"  重构项: {recon_term:.4f}")
    print(f"  KL散度项: {kl_term:.4f}")
    print(f"  ELBO: {elbo:.4f}")
    
    # 重参数化技巧演示
    print(f"\n重参数化技巧:")
    for i, (mu, sigma) in enumerate(zip(mu_encoder, sigma_encoder)):
        z, epsilon = vi.reparameterization_trick(mu, sigma)
        print(f"  维度{i+1}: z = {mu:.2f} + {sigma:.2f} × {epsilon:.3f} = {z:.4f}")
    
    print(f"\n变分推断的优势:")
    print(f"• 提供了处理不确定性的框架")
    print(f"• 使得生成模型训练成为可能")
    print(f"• 支持半监督学习和表示学习")

def manifold_learning_theory():
    """流形学习理论"""
    print("\n" + "="*60)
    print("流形学习与深度学习")
    print("="*60)
    
    print("流形假设:")
    print("• 高维数据往往分布在低维流形上")
    print("• 深度网络学习流形的表示")
    print("• 自编码器发现数据的内在维度")
    print()
    
    class ManifoldLearning:
        """流形学习算法"""
        
        def __init__(self, input_dim, manifold_dim):
            self.input_dim = input_dim
            self.manifold_dim = manifold_dim
            
        def local_linear_embedding(self, data_points, k_neighbors=3):
            """局部线性嵌入(LLE)算法的概念演示"""
            print("局部线性嵌入(LLE)步骤:")
            print("1. 找到每个点的k个最近邻")
            print("2. 计算重构权重")
            print("3. 在低维空间中保持相同的权重关系")
            
            n_points = len(data_points)
            weights = []
            
            # 简化演示：假设已知邻居
            for i in range(n_points):
                # 模拟权重计算
                w = [random.uniform(0, 1) for _ in range(k_neighbors)]
                w_sum = sum(w)
                w = [wi/w_sum for wi in w]  # 归一化
                weights.append(w)
                
                print(f"   点{i+1}的重构权重: {[f'{wi:.3f}' for wi in w]}")
            
            return weights
            
        def isometric_feature_mapping(self, distance_matrix):
            """等距映射(Isomap)的概念演示"""
            print("\n等距映射(Isomap)步骤:")
            print("1. 构建k近邻图")
            print("2. 计算所有点对的测地距离")
            print("3. 应用多维尺度变换(MDS)")
            
            n_points = len(distance_matrix)
            
            # 模拟测地距离计算
            print("测地距离矩阵:")
            for i in range(n_points):
                row = []
                for j in range(n_points):
                    if i == j:
                        geodesic_dist = 0.0
                    else:
                        geodesic_dist = distance_matrix[i][j] * random.uniform(1.0, 2.0)
                    row.append(geodesic_dist)
                print(f"   {[f'{d:.2f}' for d in row]}")
            
            return distance_matrix
    
    # 流形学习演示
    print("流形学习算法演示:")
    
    # 生成模拟高维数据点
    data_points = [
        [1.0, 2.0, 0.5, 1.5],
        [1.2, 1.8, 0.6, 1.4], 
        [2.0, 1.0, 1.2, 0.8],
        [1.8, 1.2, 1.1, 0.9]
    ]
    
    distance_matrix = [
        [0.0, 0.5, 1.8, 1.5],
        [0.5, 0.0, 1.6, 1.3],
        [1.8, 1.6, 0.0, 0.3],
        [1.5, 1.3, 0.3, 0.0]
    ]
    
    ml = ManifoldLearning(input_dim=4, manifold_dim=2)
    
    print(f"输入数据维度: {ml.input_dim}")
    print(f"流形维度: {ml.manifold_dim}")
    print(f"数据点数量: {len(data_points)}")
    
    # LLE演示
    weights = ml.local_linear_embedding(data_points)
    
    # Isomap演示  
    geodesic_distances = ml.isometric_feature_mapping(distance_matrix)
    
    print(f"\n流形学习在深度学习中的应用:")
    print(f"• 自编码器: 学习数据的紧致表示")
    print(f"• t-SNE: 高维数据可视化")
    print(f"• 生成模型: 在流形上生成新数据")

def advanced_regularization_theory():
    """高级正则化理论"""
    print("\n" + "="*60)
    print("高级正则化技术")
    print("="*60)
    
    print("正则化的理论基础:")
    print("• 贝叶斯观点: 正则化等价于先验分布")
    print("• 信息论观点: 最小描述长度原理")
    print("• 几何观点: 约束优化问题")
    print()
    
    class AdvancedRegularization:
        """高级正则化技术实现"""
        
        def spectral_normalization(self, weight_matrix, n_iterations=1):
            """谱归一化"""
            print("谱归一化原理:")
            print("• 控制权重矩阵的最大奇异值")
            print("• 确保Lipschitz约束")
            print("• 提高GAN训练稳定性")
            
            # 简化的谱归一化实现
            u = [random.gauss(0, 1) for _ in range(len(weight_matrix))]
            v = [random.gauss(0, 1) for _ in range(len(weight_matrix[0]))]
            
            for _ in range(n_iterations):
                # v = W^T u / ||W^T u||
                wt_u = [sum(weight_matrix[i][j] * u[i] for i in range(len(weight_matrix))) 
                       for j in range(len(weight_matrix[0]))]
                norm_wt_u = math.sqrt(sum(x**2 for x in wt_u))
                if norm_wt_u > 0:
                    v = [x / norm_wt_u for x in wt_u]
                
                # u = W v / ||W v||
                w_v = [sum(weight_matrix[i][j] * v[j] for j in range(len(weight_matrix[0]))) 
                      for i in range(len(weight_matrix))]
                norm_w_v = math.sqrt(sum(x**2 for x in w_v))
                if norm_w_v > 0:
                    u = [x / norm_w_v for x in w_v]
            
            # 计算谱范数 σ = u^T W v
            spectral_norm = sum(u[i] * sum(weight_matrix[i][j] * v[j] 
                                         for j in range(len(weight_matrix[0]))) 
                             for i in range(len(weight_matrix)))
            
            return spectral_norm, u, v
            
        def dropout_bayesian_interpretation(self, dropout_rate=0.5):
            """Dropout的贝叶斯解释"""
            print(f"\nDropout的贝叶斯解释:")
            print(f"• Dropout率: {dropout_rate}")
            print(f"• 等价于对权重施加先验分布")
            print(f"• 近似贝叶斯推断")
            
            # 模拟贝叶斯权重分布
            prior_precision = 1.0 / (1 - dropout_rate)
            posterior_variance = 1.0 / prior_precision
            
            print(f"• 先验精度: {prior_precision:.3f}")
            print(f"• 后验方差: {posterior_variance:.3f}")
            
            return prior_precision, posterior_variance
            
        def weight_decay_l2_regularization(self, weights, lambda_reg=0.01):
            """权重衰减与L2正则化"""
            print(f"\nL2正则化分析:")
            print(f"• 正则化参数λ = {lambda_reg}")
            
            # 计算L2范数
            l2_norm = sum(w**2 for w in weights)
            l2_penalty = 0.5 * lambda_reg * l2_norm
            
            # L2正则化梯度
            l2_gradient = [lambda_reg * w for w in weights]
            
            print(f"• L2范数: {l2_norm:.4f}")
            print(f"• L2惩罚项: {l2_penalty:.6f}")
            print(f"• 梯度修正: {[f'{g:.4f}' for g in l2_gradient[:3]]}...")
            
            return l2_penalty, l2_gradient
    
    # 正则化技术演示
    print("高级正则化技术演示:")
    
    reg = AdvancedRegularization()
    
    # 1. 谱归一化演示
    weight_matrix = [[0.8, 0.3, 0.2], [0.1, 0.9, 0.4], [0.5, 0.2, 0.7]]
    spectral_norm, u, v = reg.spectral_normalization(weight_matrix)
    
    print(f"权重矩阵: {weight_matrix}")
    print(f"谱范数: {spectral_norm:.4f}")
    
    # 2. Dropout的贝叶斯解释
    prior_prec, post_var = reg.dropout_bayesian_interpretation(0.3)
    
    # 3. L2正则化
    weights = [0.5, -0.8, 0.3, 1.2, -0.4]
    l2_penalty, l2_grad = reg.weight_decay_l2_regularization(weights, 0.01)

def main():
    """主函数"""
    print("🧮 深度学习数学原理深度解析")
    print("=" * 70)
    
    mathematical_foundations_intro()
    chain_rule_detailed_analysis()
    information_theory_in_deep_learning()
    optimization_theory_advanced()
    variational_inference_theory()
    manifold_learning_theory()
    advanced_regularization_theory()
    
    print("\n" + "=" * 70)
    print("🎯 数学理论总结")
    print()
    print("通过本模块你学到了:")
    print("• 链式法则在反向传播中的精确应用")
    print("• 信息论如何指导损失函数设计")
    print("• 高级优化算法的数学原理")
    print("• 变分推断在生成模型中的作用")
    print("• 流形学习的几何直觉")
    print("• 正则化技术的理论基础")
    print()
    print("这些数学基础是理解和发展新的深度学习")
    print("算法的关键！继续深入研究这些理论将")
    print("帮助你成为真正的深度学习专家。")

if __name__ == "__main__":
    main()