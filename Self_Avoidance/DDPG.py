import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.nn.utils.weight_norm as weight_norm
# import torch.nn.utils.parametrizations.weight_norm as weight_norm
import tools


class LayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, activation=torch.tanh, out_fn=lambda x: x):
        """
        初始化这个两层神经网络
        :param num_in: 输入的节点数量
        :param num_out: 输出的节点数量
        :param hidden_dim: 隐藏层节点数量
        :param activation: 激活函数
        :param out_fn: 输出的函数。这里使用了一个匿名函数 (lambda 函数)，该函数接受一个参数 x，并直接返回 x。
        例如，如果你创建了一个 LayerFC 类的实例时提供了一个不同的输出函数，比如 out_fn=lambda x: torch.sigmoid(x)，
        那么在前向传播过程中就会应用该函数来处理神经网络的输出。这使得 LayerFC 类更加灵活，能够适应不同的需求。
        """
        super(LayerFC, self).__init__()
        self.bn0 = nn.BatchNorm1d(num_in)
        # self.fc1 = weight_norm(torch.nn.Linear(num_in, 128))
        self.fc1 = torch.nn.Linear(num_in, 128)
        self.bn1 = nn.BatchNorm1d(128)
        # self.fc2 = weight_norm(torch.nn.Linear(128, 256))
        self.fc2 = torch.nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        # self.fc3 = weight_norm(torch.nn.Linear(256, 128))
        self.fc3 = torch.nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        # self.fc4 = weight_norm(torch.nn.Linear(128, hidden_dim))
        self.fc4 = torch.nn.Linear(128, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        # self.fc5 = weight_norm(torch.nn.Linear(hidden_dim, num_out))
        self.fc5 = torch.nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.prelu = nn.PReLU()
        # self.prelu = torch.tanh
        self.out_fn = out_fn
        # self.scale_factor = nn.Parameter(torch.tensor([random.random() for _ in range(3)]), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        # 使用 Xavier/Glorot 初始化
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        # print("输入的x：{}".format(x))
        x = self.bn0(x)
        # print("通过bn之后的x:{}".format(x))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.prelu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.prelu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        # print("self.bn4:{}".format(x))
        x = self.prelu(x)
        # print("self.prelu:{}".format(x))
        # print("self.fc5:{}".format(self.fc5(x)))
        x = self.out_fn(self.fc5(x))
        # print("输出x:{}".format(x))
        return x


class DDPG:
    """
    DDPG算法
    """
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim, discrete,
                 action_bound, sigma, actor_lr, critic_lr, tau, gamma, max_eps_episode, min_eps,
                 regularization_strength, wd, device):
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
        :param max_eps_episode:最大贪心次数
        :param min_eps:最小贪心概率
        :param regularization_strength:正则化强度
        :param wd:正则化强度
        :param device:设备
        """
        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = LayerFC(num_in_actor, num_out_actor, hidden_dim, activation=torch.tanh, out_fn=out_fn).to(device)
        self.target_actor = LayerFC(num_in_actor, num_out_actor, hidden_dim, activation=torch.tanh, out_fn=out_fn).to(device)
        self.critic = LayerFC(num_in_critic, 1, hidden_dim, activation=nn.PReLU()).to(device)
        self.target_critic = LayerFC(num_in_critic, 1, hidden_dim, activation=nn.PReLU()).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=wd)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=wd)
        self.gamma = gamma
        # 高斯噪声的标准差，均值直接设为0
        self.sigma = sigma
        self.action_bound = action_bound
        self.tau = tau
        self.action_dim = num_out_actor
        self.max_eps_episode = max_eps_episode
        self.min_eps = min_eps
        self.regularization_strength = regularization_strength
        self.device = device
        self.step = 0
        self.train = True
        # 可以被初始化的网络的字典，用于承接预训练的模型参数
        self.net_dict = {'actor': self.actor,
                         'target_actor': self.target_actor,
                         'critic': self.critic,
                         'target_critic': self.target_critic}
        # 可以被初始化的优化器的字典，用于承接优化器
        self.net_optim = {'actor': self.actor_optimizer,
                          'critic': self.critic_optimizer}

    def normal_take_action(self, state):
        """
        通用选择动作，不含贪心策略
        :param state:状态值
        :return: 动作值
        """
        action = self.actor(state).detach().cpu().numpy()
        # 给动作添加噪声，增加搜索
        # 正态分布产生随机数并乘以标准差
        action = action + self.sigma * np.random.randn(self.action_dim)
        return np.array(action)

    def take_action(self, state):
        """
        得到动作值
        :param state:状态值
        :return: 动作值
        """
        if self.max_eps_episode == 0:
            # 调用不采用贪心策略下的选取动作
            action = self.normal_take_action(state)
        else:
            # 计算贪心概率
            if self.train:
                eps = tools.epsilon_annealing(self.step, self.max_eps_episode, self.min_eps)
            else:
                eps = self.min_eps
            # 生成一个在[0,1)范围内的随机数
            sample = random.random()
            # 如果不进行随机探索或者概率大于了贪心策略概率，就不进行探索，直接通过最大的Q值来选择动作
            if sample > eps:
                # 根据Q值选择行为   Variable等效与torch.tensor
                action = self.normal_take_action(state)
            else:
                # 随机选取动作
                action = []
                for _ in range(3):
                    # 随机选择动作
                    action.append(random.random() * self.action_bound)
                # 为了配合tool做的改变
                action = np.array([action])
        # print("*" * 100)
        # print("action:{}".format(action))
        return action

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
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # 将网络设置为训练模式
        next_q_values = self.target_critic(torch.cat([next_states, self.target_actor(next_states)], dim=1))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # 计算并更新价值网络的损失
        critic_loss = torch.mean(F.mse_loss(self.critic(torch.cat([states, actions], dim=1)), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪，可以在这里添加
        # for param in self.critic.parameters():
        #     torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
        """很重要的优化器更新"""
        self.critic_optimizer.step()

        # 见笔记上的注释
        # 计算并更新策略网络的损失
        actor_loss = - torch.mean(self.critic(torch.cat([states, self.actor(states)], dim=1)))  # 策略网络为了使得Q值最大化
        # print("原始actor_loss:{}".format(actor_loss))
        actor_loss += self.regularization_strength * sum(p.pow(2).sum() for p in self.actor.parameters())
        # print("增加了正则化之后的actor_loss:{}".format(actor_loss))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 梯度裁剪，可以在这里添加
        # for param in self.actor.parameters():
        #     torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
        """很重要的优化器更新"""
        self.actor_optimizer.step()

        # 软更新策略网络
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络
        self.soft_update(self.critic, self.target_critic)


if __name__ == '__main__':
    pass
