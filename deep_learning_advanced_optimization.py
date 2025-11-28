# -*- coding: utf-8 -*-
"""
深度学习高级优化算法 - Adam、RMSprop、学习率调度

包含：Adam优化器、RMSprop、学习率衰减、批归一化、Dropout等优化技术。

注意: 此文件已作为兼容入口，推荐使用 `deep_learning.optimizers` 包。
"""

import warnings

# 转发到新包实现
from deep_learning.optimizers.advanced_optimization import *  # noqa: F401,F403

warnings.warn(
    "deep_learning_advanced_optimization.py 已迁移到 deep_learning/optimizers/ 包，"
    "请使用 deep_learning.optimizers 下的对应模块",
    DeprecationWarning,
    stacklevel=2,
)

def advanced_optimization_intro():
    """高级优化算法介绍"""
    print("=== 深度学习高级优化算法 ===")
    print("探索最新的优化技术和算法")
    print()
    print("涵盖内容:")
    print("• 二阶优化方法 (Newton, Quasi-Newton)")
    print("• 自适应优化算法 (AdamW, Lookahead, RAdam)")
    print("• 学习率调度策略 (Cosine Annealing, Warm Restart)")
    print("• 梯度修剪与梯度累积")
    print("• 分布式优化与并行训练")
    print("• 元学习与优化器学习")
    print()

def second_order_optimization():
    """二阶优化方法"""
    print("\n" + "="*70)
    print("二阶优化方法")
    print("="*70)
    
    print("牛顿法与拟牛顿法:")
    print("• 利用二阶信息加速收敛")
    print("• Hessian矩阵的计算与近似")
    print("• BFGS、L-BFGS算法")
    print("• 在深度学习中的应用挑战")
    print()
    
    class SecondOrderOptimizer:
        """二阶优化器实现"""
        
        def __init__(self, dim):
            self.dim = dim
            self.B = [[0.0 if i != j else 1.0 for j in range(dim)] for i in range(dim)]  # BFGS近似Hessian
            self.history_s = []  # 位置变化历史
            self.history_y = []  # 梯度变化历史
            
        def newton_method(self, gradient, hessian):
            """牛顿法更新步"""
            # Δx = -H^(-1) * g
            try:
                # 计算Hessian的逆矩阵 (简化实现)
                hessian_inv = self.matrix_inverse(hessian)
                update = self.matrix_vector_multiply(hessian_inv, gradient)
                return [-u for u in update]
            except:
                # 如果Hessian不可逆，退化为梯度下降
                return [-g for g in gradient]
                
        def bfgs_update(self, s, y, rho_threshold=1e-6):
            """BFGS更新近似Hessian"""
            # s = x_{k+1} - x_k, y = g_{k+1} - g_k
            
            # 计算 ρ = 1 / (y^T s)
            y_dot_s = sum(yi * si for yi, si in zip(y, s))
            
            if abs(y_dot_s) < rho_threshold:
                print(f"  警告: ρ = {y_dot_s:.2e} 过小，跳过BFGS更新")
                return
                
            rho = 1.0 / y_dot_s
            
            # BFGS更新公式
            # B_{k+1} = B_k - (B_k s s^T B_k)/(s^T B_k s) + (y y^T)/(y^T s)
            
            # 计算 B_k * s
            Bs = [sum(self.B[i][j] * s[j] for j in range(self.dim)) for i in range(self.dim)]
            
            # 计算 s^T * B_k * s  
            sBs = sum(s[i] * Bs[i] for i in range(self.dim))
            
            # 更新B矩阵
            for i in range(self.dim):
                for j in range(self.dim):
                    # 第一项: B_k
                    term1 = self.B[i][j]
                    
                    # 第二项: -(B_k s s^T B_k)/(s^T B_k s)
                    if sBs > 1e-12:
                        term2 = -(Bs[i] * Bs[j]) / sBs
                    else:
                        term2 = 0
                    
                    # 第三项: (y y^T)/(y^T s)
                    term3 = (y[i] * y[j]) * rho
                    
                    self.B[i][j] = term1 + term2 + term3
        
        def lbfgs_direction(self, gradient, m=10):
            """L-BFGS方向计算"""
            # 保持最近m次的历史信息
            if len(self.history_s) > m:
                self.history_s = self.history_s[-m:]
                self.history_y = self.history_y[-m:]
            
            if not self.history_s:
                return [-g for g in gradient]
            
            # Two-loop recursion
            alphas = []
            q = gradient[:]
            
            # 第一个循环：从新到旧
            for k in range(len(self.history_s)-1, -1, -1):
                s_k = self.history_s[k]
                y_k = self.history_y[k]
                
                rho_k = 1.0 / sum(y_k[i] * s_k[i] for i in range(self.dim))
                alpha_k = rho_k * sum(s_k[i] * q[i] for i in range(self.dim))
                
                for i in range(self.dim):
                    q[i] -= alpha_k * y_k[i]
                    
                alphas.append(alpha_k)
            
            alphas.reverse()
            
            # 初始Hessian近似
            if self.history_y:
                s_newest = self.history_s[-1]
                y_newest = self.history_y[-1]
                gamma = (sum(s_newest[i] * y_newest[i] for i in range(self.dim)) / 
                        sum(y_newest[i] * y_newest[i] for i in range(self.dim)))
                r = [gamma * qi for qi in q]
            else:
                r = q[:]
            
            # 第二个循环：从旧到新
            for k in range(len(self.history_s)):
                s_k = self.history_s[k]
                y_k = self.history_y[k]
                
                rho_k = 1.0 / sum(y_k[i] * s_k[i] for i in range(self.dim))
                beta_k = rho_k * sum(y_k[i] * r[i] for i in range(self.dim))
                
                for i in range(self.dim):
                    r[i] += s_k[i] * (alphas[k] - beta_k)
            
            return [-ri for ri in r]
        
        def matrix_inverse(self, matrix):
            """矩阵求逆 (高斯-约旦消元法)"""
            n = len(matrix)
            # 创建增广矩阵 [A | I]
            augmented = []
            for i in range(n):
                row = matrix[i][:] + [0.0] * n
                row[n + i] = 1.0
                augmented.append(row)
            
            # 高斯-约旦消元
            for i in range(n):
                # 找主元
                max_row = i
                for k in range(i + 1, n):
                    if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                        max_row = k
                
                # 交换行
                if max_row != i:
                    augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
                
                # 主元归一化
                pivot = augmented[i][i]
                if abs(pivot) < 1e-10:
                    raise ValueError("矩阵奇异，无法求逆")
                
                for j in range(2 * n):
                    augmented[i][j] /= pivot
                
                # 消元
                for k in range(n):
                    if k != i:
                        factor = augmented[k][i]
                        for j in range(2 * n):
                            augmented[k][j] -= factor * augmented[i][j]
            
            # 提取逆矩阵
            inverse = []
            for i in range(n):
                inverse.append(augmented[i][n:])
            
            return inverse
        
        def matrix_vector_multiply(self, matrix, vector):
            """矩阵向量乘法"""
            result = []
            for row in matrix:
                dot_product = sum(a * b for a, b in zip(row, vector))
                result.append(dot_product)
            return result
    
    # 二阶优化演示
    print("二阶优化算法演示:")
    
    def quadratic_function(x):
        """测试函数: f(x) = x^T A x + b^T x + c"""
        A = [[2, 1], [1, 3]]
        b = [1, -1]
        c = 0
        
        result = c
        for i in range(len(b)):
            result += b[i] * x[i]
            
        for i in range(len(A)):
            for j in range(len(A[0])):
                result += 0.5 * A[i][j] * x[i] * x[j]
        
        return result
    
    def quadratic_gradient(x):
        """梯度: g = Ax + b"""
        A = [[2, 1], [1, 3]]
        b = [1, -1]
        
        gradient = []
        for i in range(len(b)):
            gi = b[i]
            for j in range(len(A[0])):
                gi += A[i][j] * x[j]
            gradient.append(gi)
        
        return gradient
    
    def quadratic_hessian():
        """Hessian矩阵 (对于二次函数是常数)"""
        return [[2, 1], [1, 3]]
    
    optimizer = SecondOrderOptimizer(2)
    x = [1.0, 1.0]  # 初始点
    
    print(f"初始点: ({x[0]:.3f}, {x[1]:.3f})")
    print(f"初始函数值: {quadratic_function(x):.6f}")
    
    # 牛顿法优化
    for iteration in range(3):
        grad = quadratic_gradient(x)
        hess = quadratic_hessian()
        
        update = optimizer.newton_method(grad, hess)
        
        print(f"\n迭代 {iteration + 1}:")
        print(f"  梯度: ({grad[0]:.6f}, {grad[1]:.6f})")
        print(f"  更新: ({update[0]:.6f}, {update[1]:.6f})")
        
        x = [x[i] + update[i] for i in range(len(x))]
        func_val = quadratic_function(x)
        
        print(f"  新位置: ({x[0]:.6f}, {x[1]:.6f})")
        print(f"  函数值: {func_val:.10f}")
        
        # 检查收敛
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        if grad_norm < 1e-8:
            print(f"  已收敛！梯度范数: {grad_norm:.2e}")
            break

def adaptive_learning_rate_methods():
    """自适应学习率方法"""
    print("\n" + "="*70)
    print("自适应学习率优化算法")
    print("="*70)
    
    print("现代优化算法:")
    print("• AdamW: Adam + Weight Decay解耦")
    print("• RAdam: Rectified Adam")
    print("• Lookahead: 慢权重更新机制")
    print("• AdaBound: 自适应边界优化")
    print()
    
    class ModernOptimizers:
        """现代优化算法实现"""
        
        def __init__(self, params_size):
            self.params_size = params_size
            self.reset()
            
        def reset(self):
            """重置优化器状态"""
            self.m = [0.0] * self.params_size     # 一阶动量
            self.v = [0.0] * self.params_size     # 二阶动量
            self.t = 0                            # 时间步
            self.slow_params = [0.0] * self.params_size  # Lookahead慢参数
            
        def adamw_optimizer(self, params, gradients, lr=0.001, beta1=0.9, beta2=0.999, 
                           weight_decay=0.01, epsilon=1e-8):
            """AdamW优化器 - Adam + Weight Decay解耦"""
            self.t += 1
            updated_params = []
            
            for i in range(len(params)):
                # 更新动量
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (gradients[i] ** 2)
                
                # 偏置修正
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)
                
                # AdamW更新：先应用权重衰减，再应用Adam更新
                param_decayed = params[i] * (1 - lr * weight_decay)
                adam_update = lr * m_hat / (math.sqrt(v_hat) + epsilon)
                
                new_param = param_decayed - adam_update
                updated_params.append(new_param)
                
            return updated_params
            
        def radam_optimizer(self, params, gradients, lr=0.001, beta1=0.9, beta2=0.999, 
                           epsilon=1e-8):
            """RAdam (Rectified Adam) 优化器"""
            self.t += 1
            updated_params = []
            
            # 计算ρ_∞ (渐近值)
            rho_inf = 2.0 / (1 - beta2) - 1
            
            for i in range(len(params)):
                # 更新动量
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (gradients[i] ** 2)
                
                # 偏置修正
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                
                # 计算ρ_t
                rho_t = rho_inf - 2 * self.t * (beta2 ** self.t) / (1 - beta2 ** self.t)
                
                if rho_t > 4:  # 使用修正的自适应学习率
                    v_hat = self.v[i] / (1 - beta2 ** self.t)
                    # 计算修正因子
                    l_t = math.sqrt((1 - beta2 ** self.t) / v_hat)
                    r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / 
                                  ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    
                    update = lr * m_hat * r_t / (math.sqrt(v_hat) + epsilon)
                else:  # 使用无修正动量
                    update = lr * m_hat
                
                updated_params.append(params[i] - update)
                
            return updated_params
            
        def lookahead_optimizer(self, params, fast_params, fast_gradients, alpha=0.5, k=5):
            """Lookahead优化器 - 慢权重更新"""
            # 更新快权重 (可以使用任何优化器)
            updated_fast_params = self.adamw_optimizer(fast_params, fast_gradients)
            
            # 每k步更新慢权重
            if self.t % k == 0:
                print(f"    Lookahead更新 (步数: {self.t})")
                for i in range(len(params)):
                    # φ_{t+1} = φ_t + α(θ_{t+1} - φ_t)
                    self.slow_params[i] = (self.slow_params[i] + 
                                         alpha * (updated_fast_params[i] - self.slow_params[i]))
                return self.slow_params[:], updated_fast_params
            else:
                return params[:], updated_fast_params
        
        def adabound_optimizer(self, params, gradients, lr=0.001, beta1=0.9, beta2=0.999,
                              final_lr=0.1, gamma=1e-3, epsilon=1e-8):
            """AdaBound优化器 - 自适应边界"""
            self.t += 1
            updated_params = []
            
            for i in range(len(params)):
                # 更新动量
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (gradients[i] ** 2)
                
                # 偏置修正
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)
                
                # 计算自适应边界
                lower_bound = final_lr * (1 - 1 / ((1 - beta2) * self.t + 1))
                upper_bound = final_lr * (1 + 1 / ((1 - beta2) * self.t))
                
                # 计算步长
                step_size = lr / math.sqrt(v_hat + epsilon)
                step_size = max(lower_bound, min(upper_bound, step_size))
                
                # 参数更新
                update = step_size * m_hat
                updated_params.append(params[i] - update)
                
            return updated_params
    
    # 现代优化器性能比较
    print("现代优化器性能比较:")
    
    def himmelblau_function(x, y):
        """Himmelblau函数 - 多峰优化测试"""
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    def himmelblau_gradient(x, y):
        """Himmelblau函数梯度"""
        dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
        dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
        return [dx, dy]
    
    # 测试不同优化器
    optimizers_config = [
        ("AdamW", "adamw_optimizer"),
        ("RAdam", "radam_optimizer"), 
        ("AdaBound", "adabound_optimizer")
    ]
    
    initial_params = [3.0, 3.0]
    print(f"测试函数: Himmelblau函数")
    print(f"初始点: {initial_params}")
    print(f"迭代次数: 100")
    print()
    
    for name, method_name in optimizers_config:
        optimizer = ModernOptimizers(2)
        params = initial_params[:]
        
        for iteration in range(100):
            grad = himmelblau_gradient(params[0], params[1])
            
            if method_name == "adamw_optimizer":
                params = optimizer.adamw_optimizer(params, grad, lr=0.01)
            elif method_name == "radam_optimizer":
                params = optimizer.radam_optimizer(params, grad, lr=0.01)
            elif method_name == "adabound_optimizer":
                params = optimizer.adabound_optimizer(params, grad, lr=0.01)
        
        final_value = himmelblau_function(params[0], params[1])
        print(f"{name:>10}: 最终点({params[0]:7.4f}, {params[1]:7.4f}), 函数值={final_value:10.6f}")

def learning_rate_scheduling():
    """学习率调度策略"""
    print("\n" + "="*70)
    print("学习率调度策略")
    print("="*70)
    
    print("学习率调度的重要性:")
    print("• 初期：大学习率快速收敛")
    print("• 后期：小学习率精细调整")
    print("• 避免震荡和发散")
    print("• 提高最终性能")
    print()
    
    class LearningRateSchedulers:
        """学习率调度器集合"""
        
        def __init__(self, initial_lr=0.1):
            self.initial_lr = initial_lr
            self.current_step = 0
            
        def step_decay(self, drop_rate=0.5, epochs_drop=10):
            """阶段衰减"""
            epoch = self.current_step // epochs_drop
            lr = self.initial_lr * (drop_rate ** epoch)
            return lr
            
        def exponential_decay(self, decay_rate=0.95):
            """指数衰减"""
            lr = self.initial_lr * (decay_rate ** self.current_step)
            return lr
            
        def cosine_annealing(self, T_max=100, eta_min=0.0001):
            """余弦退火"""
            lr = eta_min + (self.initial_lr - eta_min) * (
                1 + math.cos(math.pi * self.current_step / T_max)) / 2
            return lr
            
        def cosine_annealing_warm_restarts(self, T_0=10, T_mult=2, eta_min=0.0001):
            """带热重启的余弦退火"""
            if self.current_step == 0:
                return self.initial_lr
                
            # 计算当前周期
            T_cur = T_0
            epoch_since_restart = self.current_step
            
            while epoch_since_restart >= T_cur:
                epoch_since_restart -= T_cur
                T_cur *= T_mult
            
            lr = eta_min + (self.initial_lr - eta_min) * (
                1 + math.cos(math.pi * epoch_since_restart / T_cur)) / 2
            
            return lr
            
        def polynomial_decay(self, max_steps=1000, power=1.0, end_lr=0.0001):
            """多项式衰减"""
            if self.current_step >= max_steps:
                return end_lr
            
            decay_factor = (1 - self.current_step / max_steps) ** power
            lr = (self.initial_lr - end_lr) * decay_factor + end_lr
            return lr
            
        def warmup_cosine(self, warmup_steps=100, total_steps=1000):
            """预热 + 余弦衰减"""
            if self.current_step < warmup_steps:
                # 线性预热
                lr = self.initial_lr * self.current_step / warmup_steps
            else:
                # 余弦衰减
                progress = (self.current_step - warmup_steps) / (total_steps - warmup_steps)
                progress = min(progress, 1.0)
                lr = 0.5 * self.initial_lr * (1 + math.cos(math.pi * progress))
            
            return lr
        
        def step(self):
            """更新步数"""
            self.current_step += 1
    
    # 学习率调度演示
    print("学习率调度策略演示:")
    
    scheduler = LearningRateSchedulers(initial_lr=0.1)
    total_steps = 200
    
    # 收集不同调度策略的学习率
    schedules = {
        "阶段衰减": [],
        "指数衰减": [],
        "余弦退火": [],
        "余弦热重启": [],
        "预热+余弦": []
    }
    
    for step in range(total_steps):
        scheduler.current_step = step
        
        schedules["阶段衰减"].append(scheduler.step_decay())
        schedules["指数衰减"].append(scheduler.exponential_decay())
        schedules["余弦退火"].append(scheduler.cosine_annealing(T_max=total_steps))
        schedules["余弦热重启"].append(scheduler.cosine_annealing_warm_restarts())
        schedules["预热+余弦"].append(scheduler.warmup_cosine(total_steps=total_steps))
    
    # 显示关键步数的学习率
    key_steps = [0, 20, 50, 100, 150, 199]
    print(f"{'策略':>12} | " + " | ".join(f"步数{s:>3}" for s in key_steps))
    print("-" * (12 + len(key_steps) * 9))
    
    for name, lr_values in schedules.items():
        lr_at_steps = [lr_values[s] for s in key_steps]
        print(f"{name:>12} | " + " | ".join(f"{lr:>6.4f}" for lr in lr_at_steps))
    
    print(f"\n学习率调度策略选择指南:")
    print(f"• 阶段衰减: 简单有效，需要手动设置衰减点")
    print(f"• 指数衰减: 平滑衰减，但可能衰减过快")
    print(f"• 余弦退火: 自然的衰减曲线，广泛使用")
    print(f"• 余弦热重启: 避免局部最优，适合长训练")
    print(f"• 预热+余弦: 现代训练的标准配置")

def gradient_clipping_and_accumulation():
    """梯度裁剪与梯度累积"""
    print("\n" + "="*70)
    print("梯度裁剪与梯度累积")
    print("="*70)
    
    print("梯度裁剪的必要性:")
    print("• 防止梯度爆炸")
    print("• 稳定训练过程")
    print("• 特别重要于RNN训练")
    print()
    
    print("梯度累积的应用:")
    print("• 模拟大批量训练")
    print("• 节省显存资源")
    print("• 提高训练稳定性")
    print()
    
    class GradientProcessing:
        """梯度处理工具"""
        
        def __init__(self):
            self.accumulated_gradients = []
            self.accumulation_steps = 0
            
        def gradient_clipping_norm(self, gradients, max_norm=1.0):
            """按范数裁剪梯度"""
            # 计算梯度的L2范数
            grad_norm = math.sqrt(sum(g**2 for g in gradients))
            
            if grad_norm > max_norm:
                # 缩放梯度
                scale_factor = max_norm / grad_norm
                clipped_gradients = [g * scale_factor for g in gradients]
                
                print(f"  梯度裁剪: 原范数={grad_norm:.4f}, 裁剪后={max_norm:.4f}")
                return clipped_gradients, True
            else:
                return gradients[:], False
                
        def gradient_clipping_value(self, gradients, max_value=0.5):
            """按值裁剪梯度"""
            clipped_gradients = []
            clipped_count = 0
            
            for g in gradients:
                if g > max_value:
                    clipped_gradients.append(max_value)
                    clipped_count += 1
                elif g < -max_value:
                    clipped_gradients.append(-max_value)
                    clipped_count += 1
                else:
                    clipped_gradients.append(g)
            
            if clipped_count > 0:
                print(f"  值裁剪: {clipped_count} 个梯度被裁剪到 ±{max_value}")
                
            return clipped_gradients, clipped_count > 0
            
        def gradient_accumulation(self, gradients, accumulation_steps=4):
            """梯度累积"""
            if len(self.accumulated_gradients) == 0:
                self.accumulated_gradients = [0.0] * len(gradients)
            
            # 累积当前梯度
            for i in range(len(gradients)):
                self.accumulated_gradients[i] += gradients[i]
            
            self.accumulation_steps += 1
            
            # 检查是否到达累积步数
            if self.accumulation_steps >= accumulation_steps:
                # 计算平均梯度
                averaged_gradients = [g / accumulation_steps for g in self.accumulated_gradients]
                
                # 重置累积状态
                self.accumulated_gradients = [0.0] * len(gradients)
                self.accumulation_steps = 0
                
                return averaged_gradients, True  # 返回平均梯度和更新标志
            else:
                return None, False  # 还未达到更新条件
        
        def adaptive_gradient_clipping(self, gradients, parameters, percentile=10):
            """自适应梯度裁剪"""
            # 计算参数的范数
            param_norm = math.sqrt(sum(p**2 for p in parameters))
            grad_norm = math.sqrt(sum(g**2 for g in gradients))
            
            if param_norm == 0 or grad_norm == 0:
                return gradients[:]
            
            # 根据参数范数自适应调整裁剪阈值
            max_norm = param_norm * percentile / 100.0
            
            if grad_norm > max_norm:
                scale_factor = max_norm / grad_norm
                clipped_gradients = [g * scale_factor for g in gradients]
                
                print(f"  自适应裁剪: 参数范数={param_norm:.4f}, 梯度范数={grad_norm:.4f} -> {max_norm:.4f}")
                return clipped_gradients
            else:
                return gradients[:]
    
    # 梯度处理演示
    print("梯度处理技术演示:")
    
    grad_processor = GradientProcessing()
    
    # 模拟训练过程中的梯度
    training_gradients = [
        [0.1, 0.2, -0.15],      # 正常梯度
        [2.5, -1.8, 3.2],      # 爆炸梯度
        [0.05, 0.08, -0.03],   # 小梯度
        [1.2, -0.9, 1.5],      # 中等梯度
        [0.3, 0.1, -0.2],      # 正常梯度
    ]
    
    parameters = [1.0, 0.5, -0.8]  # 模拟参数
    
    print(f"原始梯度序列:")
    for i, grad in enumerate(training_gradients):
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        print(f"  步骤{i+1}: {grad} (范数: {grad_norm:.4f})")
    
    print(f"\n1. 范数梯度裁剪 (max_norm=1.0):")
    for i, grad in enumerate(training_gradients):
        clipped_grad, was_clipped = grad_processor.gradient_clipping_norm(grad, max_norm=1.0)
        if was_clipped:
            print(f"  步骤{i+1}: {[f'{g:.4f}' for g in clipped_grad]}")
        else:
            print(f"  步骤{i+1}: 未裁剪")
    
    print(f"\n2. 自适应梯度裁剪:")
    for i, grad in enumerate(training_gradients):
        clipped_grad = grad_processor.adaptive_gradient_clipping(grad, parameters)
        original_norm = math.sqrt(sum(g**2 for g in grad))
        clipped_norm = math.sqrt(sum(g**2 for g in clipped_grad))
        if abs(original_norm - clipped_norm) > 1e-6:
            print(f"  步骤{i+1}: 范数 {original_norm:.4f} -> {clipped_norm:.4f}")
    
    print(f"\n3. 梯度累积演示 (累积4步):")
    grad_processor = GradientProcessing()  # 重置
    for i, grad in enumerate(training_gradients):
        avg_grad, should_update = grad_processor.gradient_accumulation(grad, accumulation_steps=2)
        
        if should_update:
            print(f"  累积完成，平均梯度: {[f'{g:.4f}' for g in avg_grad]}")
        else:
            print(f"  步骤{i+1}: 累积中... (已累积{grad_processor.accumulation_steps}步)")

def main():
    """主函数"""
    print("⚡ 深度学习高级优化算法")
    print("=" * 70)
    
    advanced_optimization_intro()
    second_order_optimization()
    adaptive_learning_rate_methods()
    learning_rate_scheduling()
    gradient_clipping_and_accumulation()
    
    print("\n" + "=" * 70)
    print("🎯 高级优化技术总结")
    print()
    print("掌握的优化技术:")
    print("• 二阶优化方法：牛顿法、BFGS、L-BFGS")
    print("• 现代自适应算法：AdamW、RAdam、Lookahead")
    print("• 学习率调度：余弦退火、预热策略")
    print("• 梯度处理：裁剪、累积、自适应技术")
    print()
    print("这些高级优化技术是训练大型深度学习")
    print("模型的关键工具，掌握它们将显著提升")
    print("你的模型训练效果和效率！")

if __name__ == "__main__":
    main()
