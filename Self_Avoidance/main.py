"""
该文件是无人机自主避障的main文件
需要和其他文件，如算法，环境等进行协作
先尝试使用DDPG算法，是深度确定性策略梯度算法，因此输出的动作是确定的
不需要构建概率密度分布之后再采样
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import tools
from DDPG import DDPG
import environment
# import matplotlib
# matplotlib.use('TkAgg')  # 或者其他后端


if __name__ == '__main__':
    # 是否不加载权重，重新开始训练
    retrain = True
    print("hello")
    # pth文件保存的位置
    pth_load = {'actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\actor.pth',
                'critic': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\critic.pth',
                'target_actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_actor.pth',
                'target_critic': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_critic.pth'}
    # pth_load = {'actor': r'D:\ldt\Drone_self_avoidance\Self_Avoidance\actor.pth',
    #             "critic_1": r'D:\ldt\Drone_self_avoidance\Self_Avoidance\critic_1.pth',
    #             "critic_2": r'D:\ldt\Drone_self_avoidance\Self_Avoidance\critic_2.pth',
    #             'target_critic_1': r'D:\ldt\Drone_self_avoidance\Self_Avoidance\target_critic_1.pth',
    #             'target_critic_2': r'D:\ldt\Drone_self_avoidance\Self_Avoidance\target_critic_2.pth'}
    # 策略网络学习率
    actor_lr = 1e-3
    # 价值网络学习率
    critic_lr = 1e-3
    # 迭代次数
    num_episodes = 10000
    # 隐藏节点，先暂定64，后续可以看看效果
    hidden_dim = 128
    # 折扣因子
    gamma = 0.99
    # 软更新参数 原来为0.005
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
    state_dim = 744
    # 最大贪心次数，为0是直接根据Q值来选取的动作
    max_eps_episode = 0
    # 最小贪心概率
    min_eps = 0
    # 暂定直接控制智能体的位移，所以是三维的
    action_dim = 3
    # 每一次迭代中，无人机的数量
    num_uavs = 30
    # 无人机可控风速
    v0 = 40
    # 设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:{}".format(device))

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
                 action_bound, sigma, actor_lr, critic_lr, tau, gamma, max_eps_episode, min_eps, device)
    return_list = tools.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, pth_load, retrain,
                                               max_eps_episode, min_eps, action_bound, device)

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