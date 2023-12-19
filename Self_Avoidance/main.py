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
    # pth文件保存的位置
    pth_load = {'actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\actor.pth',
                'critic': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\critic.pth',
                'target_actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_actor.pth',
                'target_critic': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_critic.pth'}
    # 策略网络学习率
    actor_lr = 5e-4
    # 价值网络学习率
    critic_lr = 5e-3
    # 迭代次数
    num_episodes = 10000
    # 隐藏节点，先暂定64，后续可以看看效果
    hidden_dim = 64
    # 折扣因子
    gamma = 0.98
    # 软更新参数
    tau = 0.005
    # 经验回放池大小
    buffer_size = 10000
    # 经验回放池最小经验数目
    minimal_size = 100
    # 每一批次选取的经验数量
    batch_size = 64
    # 高斯噪声标准差
    sigma = 0.01
    # 三维环境下动作，加上一堆状态的感知，目前是124+16=140个
    state_dim = 140
    # 暂定直接控制智能体的位移，所以是三维的
    action_dim = 3
    # 每一次迭代中，无人机的数量
    num_uavs = 30
    # 无人机可控风速
    v0 = 40
    # 设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 智能体的半径（先暂时定义为球体）
    agent_r = 1
    # 动作区域
    action_area = np.array([[0, 0, 0], [100, 100, 25]])
    # 动作最大值
    action_bound = 0.5

    # 实例化交互环境
    env = environment.Environment(agent_r, action_area, num_uavs, v0)
    # 实例化经验回放池
    replay_buffer = tools.ReplayBuffer(buffer_size)

    # 实例化DDPG对象，其实动作为非离散，所以为False
    agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False,
                 action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return_list = tools.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, pth_load)

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