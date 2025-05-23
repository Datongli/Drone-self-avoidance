import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import torch.nn.init as init
import tools
from torch.nn.utils.parametrizations import weight_norm


class PolicyNetContinuous(torch.nn.Module):
    """
    定义策略网络，由于处理的是与连续动作交互的环境，
    策略网络输出一个高斯分布的均值和标准差来表示动作分布
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, activation=F.relu):
        """
        初始化策略网络
        :param state_dim: 状态的纬度
        :param hidden_dim: 隐藏层数量
        :param action_dim: 动作的纬度
        :param action_bound: 动作的最大值
        :parameter activation: 激活函数
        """
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.activation_bound = action_bound
        self.activation = activation
        self.bn = nn.BatchNorm1d(hidden_dim)
        # 定义可以训练的均值和方差，用于在输入时归一化输入，有利于训练
        self.policy_mean = nn.Parameter(torch.zeros((state_dim,)), requires_grad=True)
        self.policy_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=True)
        # 初始化参数
        # self.init_weights()

    def init_weights(self):
        # 使用 Xavier/Glorot 初始化
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_mu, self.fc_std, self.fc5, self.fc6]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.normal_(layer.bias, mean=0, std=1)

    def input_norm(self, x):
        """
        用于归一化输入，使用的mean和std均是可以训练的参数
        :param x:输入的数据
        :return:归一化之后的输入
        """
        return (x - self.policy_mean) / self.policy_std

    def forward(self, x):
        """
        前向传播函数
        :param x:输入的值
        :return: 输出动作，对数概率密度
        """
        x = self.fc1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc5(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc6(x)
        x = self.bn(x)
        x = self.activation(x)
        mu = self.fc_mu(x)
        # 使用Softplus激活函数处理第三个全连接层的输出，得到标准差
        std = F.softplus(self.fc_std(x))
        # 创建正态分布对象，以均值和标准差作为参数
        dist = Normal(mu, std)
        # 从正态分布中进行重参数化采样，以获得动作
        normal_sample = dist.rsample()
        # 计算正态分布的对数概率密度
        log_prob = dist.log_prob(normal_sample)
        # 将采样得到的值通过tanh函数映射到[-1, 1]范围
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # 将动作缩放到合适的范围
        action = action * self.activation_bound
        # 返回动作和对数概率（熵） 这里将三个维度上的动作的熵进行了加和
        return action, torch.sum(log_prob, dim=1).view(-1, 1)


class QValueNetContinuous(torch.nn.Module):
    """
    定义价值网络
    价值网络的输入是状态和动作的拼接向量，输出一个实数来表示动作价值
    """

    def __init__(self, state_dim, hidden_dim, action_dim, activation=F.relu):
        """
        初始化动作价值函数
        :param state_dim: 状态纬度
        :param hidden_dim: 隐藏纬度
        :param action_dim: 动作纬度
        :param activation: 激活函数
        """
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = activation
        # 定义可以训练的均值和方差，用于在输入时归一化输入，有利于训练
        self.value_mean = nn.Parameter(torch.zeros((state_dim + action_dim,)), requires_grad=True)
        self.value_std = nn.Parameter(torch.ones((state_dim + action_dim,)), requires_grad=True)
        # self.init_weights()

    def init_weights(self):
        # 使用 Xavier/Glorot 初始化
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_out, self.fc5, self.fc6]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.normal_(layer.bias, mean=0, std=1)

    def input_norm(self, x):
        """
        用于将输入归一化的类函数，所使用的mean和std均为可以训练的参数
        :param x: 输入数据
        :return: 输出的归一化之后的数据
        """
        return (x - self.value_mean) / self.value_std

    def forward(self, x, a):
        """
        价值网络前向传播函数
        :param x: 状态值
        :param a: 相应状态下的动作值
        :return: 判断的价值
        """
        x = torch.cat([x, a], dim=1)
        x = self.fc1(x)
        # x = self.bn(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.bn(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.bn(x)
        x = self.activation(x)
        x = self.fc4(x)
        # x = self.bn(x)
        x = self.activation(x)
        x = self.fc5(x)
        # x = self.bn(x)
        x = self.activation(x)
        x = self.fc6(x)
        # x = self.bn(x)
        x = self.activation(x)
        return self.fc_out(x)


class SACContinuous:
    """处理连续动作的SAC算法"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, max_eps_episode, min_eps, wd, device):
        """
        初始化SAC算法
        :param state_dim:状态纬度
        :param hidden_dim:隐藏纬度
        :param action_dim:动作纬度
        :param action_bound:动作最大限制
        :param actor_lr:策略网络的学习率
        :param critic_lr:价值网络的学习率
        :param alpha_lr:
        :param target_entropy:目标熵
        :param tau:软更新的参数
        :param gamma:折扣因子
        :param max_eps_episode:最大贪心次数
        :param min_eps:最小贪心概率
        :param wd: 正则化强度
        :param device:设备
        """
        # 策略网络
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound, activation=nn.PReLU()).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim, activation=nn.PReLU()).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim, activation=nn.PReLU()).to(device)
        # 第一个目标Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim, activation=nn.PReLU()).to(device)
        # 第二个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim, activation=nn.PReLU()).to(device)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 初始化策略网络和价值网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=wd)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, weight_decay=wd)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, weight_decay=wd)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.5)
        self.critic_1_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_1_optimizer, step_size=500, gamma=0.5)
        self.critic_2_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_2_optimizer, step_size=500, gamma=0.5)
        # 使用alpha的log值，可以使训练结果比较稳定， 这里默认alpha为0.01（初始值）
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # 可以对alpha求梯度
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        # 目标熵的大小
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.max_eps_episode = max_eps_episode
        self.min_eps = min_eps
        self.device = device
        self.step = 0
        self.activation_bound = action_bound
        self.train = True
        # 可以被初始化的网络的字典，用于承接预训练的模型参数
        self.net_dict = {'SAC_actor': self.actor,
                         "critic_1": self.critic_1,
                         "critic_2": self.critic_2,
                         'target_critic_1': self.target_critic_1,
                         'target_critic_2': self.target_critic_2}
        # 可以被初始化的优化器的字典，用于承接优化器
        self.net_optim = {'actor': self.actor_optimizer,
                          'critic_1': self.critic_1_optimizer,
                          'critic_2': self.critic_2_optimizer}

    def normal_take_action(self, state):
        """
        通用选择动作，不使用贪心策略
        :param state: 状态值
        :return: 动作值
        """
        # 采取动作
        action = self.actor(state)[0].detach().cpu().numpy()
        # 提取动作张量中的数值，并将其作为单一元素构成的列表返回
        return np.array(action)

    def take_action(self, state):
        """
        采取动作的函数,使用贪心策略
        :param state: 状态值
        :return: 采取的动作
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
                # 让无人机在目标的方向上前进距离为action_bound
                uav_local = np.array(state[0][:3].cpu())
                target_local = np.array(state[0][6:9].cpu())
                # 计算两个点之间的向量
                vector = target_local - uav_local
                # 将向量标准化为单位向量
                normalized_vector = vector / np.linalg.norm(vector)
                # 将向量缩放为期望的距离
                desired_distance = self.activation_bound
                action = normalized_vector * desired_distance
                # 为了配合tool做的改变
                action = np.array([action])
        return action

    def calc_target(self, rewards, next_states, dones):
        """
        计算目标Q值
        :param rewards: 奖励值
        :param next_states: 下一状态值
        :param dones: 是否完成
        :return: 目标Q值
        """
        # 得到下一个动作和熵
        next_actions, log_prob = self.actor(next_states)
        # 方便计算
        entropy = - log_prob
        # 得到两个Q值
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        # min(Q) - alpha * log(\pi)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        # 这里的gamma和书上的相乘不太一样，范围扩大了
        td_target = rewards + self.gamma * next_value * (1 - dones)
        # 得到目标Q的值
        return td_target

    def soft_update(self, net, target_net):
        """
        目标网络参数软更新
        :param net: 策略网络或者价值网络
        :param target_net: 目标（策略或者价值）网络
        :return:
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        """
        用于更新网络参数的函数
        :param transition_dict:中间变量字典
        :return:
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        print("td_target:{}".format(td_target))
        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = - log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(- self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        print("self.log_alpha.exp():{}".format(self.log_alpha.exp()))
        print("entropy:{}".format(entropy))
        print("q1_value:{}".format(q1_value))
        print("q2_value:{}".format(q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        """检查一下各个loss的情况，看看是否梯度爆炸"""
        print("critic_1_loss:{}".format(critic_1_loss))
        print("critic_2_loss:{}".format(critic_2_loss))
        print("actor_loss:{}".format(actor_loss))
        print("alpha_loss:{}".format(alpha_loss))


if __name__ == '__main__':
    pass
