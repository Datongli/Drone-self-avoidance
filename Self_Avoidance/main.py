"""
该文件是无人机自主避障的main文件
需要和其他文件，如算法，环境等进行协作
先尝试使用DDPG算法，是深度确定性策略梯度算法，因此输出的动作是确定的
不需要构建概率密度分布之后再采样
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import tools
from DDPG import DDPG
import environment
import random


if __name__ == '__main__':
    # 策略网络学习率
    actor_lr = 5e-4
    # 价值网络学习率
    critic_lr = 5e-3
    # 迭代次数
    num_episodes = 1000
    # 隐藏节点
    hidden_dim = 64
    # 折扣因子
    gamma = 0.98
    # 软更新参数
    tau = 0.005
    # 经验回放池大小
    buffer_size = 10000
    # 经验回放池最小经验数目
    minimal_size = 1000
    # 每一批次选取的经验数量
    batch_size = 64
    # 高斯噪声标准差
    sigma = 0.01
    # 三维环境下动作
    state_dim = 3
    # 暂定直接控制智能体的位移，所以是三维的
    action_dim = 3
    # 设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    """生成一下障碍物:球体(半球)，立方体"""
    obstacle = {"sphere": ([[20, 30, 0], 16], [[70, 25, 60], 18]),
                "cube": ([[40, 40, 0], [50, 50, 100]],
                         [[5, 80, 0], [95, 95, 50]],
                         [[85, 5, 0], [95, 70, 40]])}
    # 智能体的初始化位置
    agent_state = np.array([10.0, 10.0, 15.0])
    # 智能体的半径（先暂时定义为球体）
    agent_r = 1
    # 目标区域(对角线)
    target_area = np.array([[92, 92, 92], [98, 98, 98]])
    # 动作区域
    action_area = np.array([[0, 0, 0], [100, 100, 100]])
    # 动作最大值
    action_bound = 0.05

    # 实例化交互环境
    env = environment.Environment(obstacle, agent_state, agent_r, target_area, action_area)
    random.seed(0)
    torch.manual_seed(0)
    # 实例化经验回放池
    replay_buffer = tools.ReplayBuffer(buffer_size)

    # 实例化DDPG对象，其实动作为非离散，所以为False
    agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False,
                 action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return_list = tools.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    # 绘图
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG')
    plt.show()

    mv_return = tools.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG')
    plt.show()