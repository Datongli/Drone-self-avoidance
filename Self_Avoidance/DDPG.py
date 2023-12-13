import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
    """
    策略网络
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_boud):
        """
        初始化策略网络
        :param state_dim:状态维度
        :param hidden_dim: 隐藏层节点
        :param action_dim: 动作维度
        :param action_boud: 环境可以接受的动作最大值
        """
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        # action_boud：环境可以接受的动作最大值
        self.action_boud = action_boud

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 返回归一化到[-1,1]之间的神经网络输出，乘以可以接受动作的最大值，相当于乘以了一个基准
        return F.tanh(self.fc2(x)) * self.action_boud


class QValueNet(torch.nn.Module):
    """
    Q价值网络
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        初始化Q价值网络
        :param state_dim:状态维度
        :param hidden_dim: 隐藏节点
        :param action_dim: 动作维度
        """
        super(QValueNet, self).__init__()
        # Q网络输入的是状态和动作拼接后的向量
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # 输出的是一个值，表示该状态动作对的价值
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # 拼接状态值和动作值
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(torch.nn.Module):
    """
    这是一个简单的两层神经网络
    """
    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x:x):
        """
        初始化这个两层神经网络
        :param num_in: 输入的节点数量
        :param num_out: 输出的节点数量
        :param hidden_dim: 隐藏层节点数量
        :param activation: 激活函数
        :param out_fn: 输出的函数。这里使用了一个匿名函数 (lambda 函数)，该函数接受一个参数 x，并直接返回 x。
        例如，如果你创建了一个 TwoLayerFC 类的实例时提供了一个不同的输出函数，比如 out_fn=lambda x: torch.sigmoid(x)，
        那么在前向传播过程中就会应用该函数来处理神经网络的输出。这使得 TwoLayerFC 类更加灵活，能够适应不同的需求。
        """
        super(TwoLayerFC, self).__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # 通过最后一层的处理，可以自定义输出
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    """
    DDPG算法
    """
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim, discrete,
                 action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        """
        初始化DDPG算法
        :param num_in_actor:策略网络输入节点（状态维度）
        :param num_out_actor:策略网络输出节点（动作维度）
        :param num_in_critic:价值网络输入节点
        :param hidden_dim:隐藏层节点
        :param discrete:是否为离散
        :param action_bound:环境可以接受的动作最大值
        :param sigma:高斯噪声的标准差
        :param actor_lr:策略网络学习率
        :param critic_lr:价值网络学习率
        :param tau:更新目标网络的参数值
        :param gamma:折扣因子
        :param device:设备
        """
        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(num_in_actor, num_out_actor, hidden_dim, activation=F.relu, out_fn=out_fn).to(device)
        self.target_actor = TwoLayerFC(num_in_actor, num_out_actor, hidden_dim, activation=F.relu, out_fn=out_fn).to(device)
        self.critic = TwoLayerFC(num_in_critic, 3, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 3, hidden_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        # 高斯噪声的标准差，均值直接设为0
        self.sigma = sigma
        self.action_bound = action_bound
        self.tau = tau
        self.action_dim = num_out_actor
        self.device = device

    def take_action(self, state):
        """
        得到动作值
        :param state:状态值
        :return: 动作值
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        # 给动作添加噪声，增加搜索
        # 正态分布产生随机数并乘以标准差
        action = action + self.sigma * np.random.randn(self.action_dim)
        return np.array(action)

    def soft_update(self, net, target_net):
        """
        软更新参数
        :param net:使用的网络
        :param target_net:要进行软更新的目标网络
        :return:
        """
        # 使用zip()函数同时迭代两个网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        """
        参数更新
        :param transition_dict: 过度态字典
        :return:
        """
        # 从过渡态字典中得到相应的值
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 见笔记上的注释
        next_q_values = self.target_critic(torch.cat([next_states, self.target_actor(next_states)], dim=1))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # 计算并更新价值网络的损失
        critic_loss = torch.mean(F.mse_loss(self.critic(torch.cat([states, actions], dim=1)), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 见笔记上的注释
        # 计算并更新策略网络的损失
        actor_loss = - torch.mean(self.critic(torch.cat([states, self.actor(states)], dim=1)))  # 策略网络为了使得Q值最大化
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络
        self.soft_update(self.critic, self.target_critic)


if __name__ == '__main__':
    pass







