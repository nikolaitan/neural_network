import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pandas as pd  # 数据处理
import requests    # 网络请求（用于获取真实数据）

# 设置随机种子保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------
# 中文注释：系统辨识模块
# Russian: Модуль системной идентификации
# ---------------------------
def load_real_data(url):
    """从网络加载真实数据（示例使用模拟数据）"""
    # 实际使用时替换为真实数据URL
    # response = requests.get(url)
    # return pd.read_csv(response.content)
    
    # 模拟数据生成
    t = np.linspace(0, 10, 1000)
    K = 1.5   # 增益系数
    Tm = 0.05 # 时间常数
    noise = 0.005 * np.max(K) * np.random.randn(len(t))  # 0.5%噪声
    y = K / ((Tm * t + 1) * t) + noise
    return t, y

class SystemIdentifier:
    def __init__(self, data):
        self.data = data
        
    def estimate_tf(self):
        """使用最小二乘法辨识传递函数参数"""
        # 这里简化实现，实际应用推荐使用专业工具箱
        # 示例返回固定参数
        return {'K': 1.5, 'Tm': 0.05}

# ---------------------------
# 中文注释：PID控制器模块
# Russian: Модуль PID-регулятора
# ---------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.integral = 0
        self.prev_error = 0
        
    def compute(self, error, dt):
        """计算PID输出"""
        derivative = (error - self.prev_error) / dt
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.output_limits[1], self.output_limits[1])
        
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        self.prev_error = error
        return output

# ---------------------------
# 中文注释：DDPG智能体模块
# Russian: Модуль DDPG агента
# ---------------------------
class Actor(nn.Module):
    """策略网络（Actor）"""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Q值网络（Critic）"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.buffer = deque(maxlen=1e5)
        self.max_action = max_action
        
    def select_action(self, state):
        """根据当前状态选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.reshape(1, -1))
            return self.actor(state_tensor).numpy().flatten()
    
    def update(self, batch_size=64, gamma=0.99, tau=1e-3):
        """更新网络参数"""
        if len(self.buffer) < batch_size:
            return
        
        # 从经验池采样
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Critic更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + gamma * self.critic_target(next_states, next_actions) * (1 - dones)
            
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# ---------------------------
# 中文注释：主程序入口
# Russian: Точка входа в программу
# ---------------------------
if __name__ == "__main__":
    # 参数配置
    state_dim = 1       # 状态维度（当前角度误差）
    action_dim = 1      # 动作维度（控制扭矩）
    max_action = 3.5    # 最大允许扭矩(N·m)
    buffer_size = 1e5   # 经验池容量
    batch_size = 64     # 批次大小
    gamma = 0.99        # 折扣因子
    tau = 1e-3          # 目标网络更新率
    lr_actor = 1e-4     # Actor学习率
    lr_critic = 1e-3    # Critic学习率
    
    # 数据准备
    url = "https://example.com/real_data.csv"  # 替换为真实数据链接
    t, y_measured = load_real_data(url)
    
    # 系统辨识
    identifier = SystemIdentifier((t, y_measured))
    params = identifier.estimate_tf()
    print(f"Identified parameters: K={params['K']:.3f}, Tm={params['Tm']:.3f}")
    
    # PID整定（Ziegler-Nichols方法）
    Ku = 1.2  # 临界增益（需实验测定）
    Pu = 2.0  # 振荡周期（需实验测定）
    pid = PIDController(
        Kp=0.6*Ku,
        Ki=1.2*Ku/Pu,
        Kd=0.075*Ku*Pu,
        output_limits=(-max_action, max_action)
    )
    
    # DDPG训练
    agent = DDPGAgent(state_dim, action_dim, max_action)
    scores = []
    
    for episode in range(100):
        state = np.random.uniform(-1, 1)  # 归一化初始状态
        episode_reward = 0
        
        for step in range(1000):
            action = agent.select_action(state)
            
            # 模拟环境反馈（需替换为真实系统交互）
            # next_state = real_system_step(action)
            next_state = np.random.uniform(-1, 1)
            reward = -np.sum((next_state - state)**2)  # 简化奖励函数
            
            agent.buffer.append((state, action, reward, next_state, False))
            state = next_state
            episode_reward += reward
            
            if step % UPDATE_EVERY == 0:
                agent.update(batch_size=batch_size, gamma=gamma, tau=tau)
                
        scores.append(episode_reward)
        print(f"Episode {episode+1}\tReward: {episode_reward:.2f}")
        
    # 结果可视化
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='DDPG Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('DDPG Training Progress')
    plt.legend()
    plt.show()