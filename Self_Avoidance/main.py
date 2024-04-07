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
from SAC import SACContinuous
import environment
# 可以显示中文
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


if __name__ == '__main__':
    # 是否不加载权重，重新开始训练
    retrain = False
    # 选择训练模型是DDPG还是SAC
    train_model = 'DDPG'
    # 策略网络学习率
    actor_lr = 1e-3
    # 价值网络学习率
    critic_lr = 1e-3
    # SAC模型中的alpha参数学习率
    alpha_lr = 1e-5
    # 迭代次数
    num_episodes = 2000
    # 隐藏节点，先暂定64，后续可以看看效果
    hidden_dim = 64
    # 折扣因子
    gamma = 0.99
    # 软更新参数 原来为0.005
    tau = 0.05
    # 初始化环境难度的等级
    level = 9
    # 每一批次选取的经验数量
    batch_size = 128
    # 经验回放池大小
    buffer_size = 1000000
    # 经验回放池最小经验数目
    minimal_size = batch_size
    # 高斯噪声标准差
    sigma = 0.01
    # 状态纬度，目前是无人机三维坐标，三维坐标变化量，目标点三维坐标
    num_in_actor = 21
    # 最大贪心次数，为0是直接根据Q值来选取的动作
    # 想要提升模型的性能，最好把训练的侧重点放在模型上
    max_eps_episode = 0
    # 最小贪心概率
    min_eps = 0
    # 正则化强度
    wd = 0.001
    # 暂定直接控制智能体的位移，所以是三维的
    action_dim = 3
    # 目标熵，用于SAC算法
    target_entropy = - action_dim
    # 每一次迭代中，无人机的数量
    num_uavs = 15
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
    action_bound = 2.0

    # 实例化交互环境
    env = environment.Environment(agent_r, action_area, num_uavs, v0)
    env.level = level
    # 实例化经验回放池
    replay_buffer = tools.ReplayBuffer(buffer_size)
    # 保留选取的地址
    bn_txt = r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\all_bn.txt'

    # 实例化智能体对象，可以选择使用的训练模型
    if train_model == 'DDPG':
        # 实例化DDPG对象，其实动作为非离散，所以为False
        agent = DDPG(num_in_actor, action_dim, num_in_actor + action_dim, hidden_dim, False,
                     action_bound, sigma, actor_lr, critic_lr, tau, gamma, max_eps_episode, min_eps,
                     wd, device)
        pth_load = {'actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\actor.pth',
                    'critic': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\critic.pth',
                    'target_actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_actor.pth',
                    'target_critic': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_critic.pth'}
        # 寻迹和避障的初始权重
        # actor_pth_load = {'actor_tracing': r'C:\Users\Lenovo\Desktop\训练权重保存\本科毕设\DDPG\只寻迹\参数2\actor_tracing.pth',
        #                   'actor_avoid': r'C:\Users\Lenovo\Desktop\训练权重保存\本科毕设\DDPG\只避障\actor_avoid.pth',
        #                   'target_actor_tracing': r'C:\Users\Lenovo\Desktop\训练权重保存\本科毕设\DDPG\只寻迹\参数2\target_actor_tracing.pth',
        #                   'target_actor_avoid': r'C:\Users\Lenovo\Desktop\训练权重保存\本科毕设\DDPG\只避障\target_actor_avoid.pth'}
        actor_pth_load = {
            'actor_tracing': r'C:\Users\Lenovo\Desktop\训练权重保存\本科毕设\DDPG\只寻迹\参数2\actor_tracing.pth',
            'target_actor_tracing': r'C:\Users\Lenovo\Desktop\训练权重保存\本科毕设\DDPG\只寻迹\参数2\target_actor_tracing.pth'}
    if train_model == 'SAC':
        agent = SACContinuous(num_in_actor, hidden_dim, action_dim, action_bound, actor_lr, critic_lr,
                              alpha_lr, target_entropy, tau, gamma, max_eps_episode, min_eps, wd, device)
        pth_load = {'SAC_actor': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\SAC_actor.pth',
                    "critic_1": r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\critic_1.pth',
                    "critic_2": r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\critic_2.pth',
                    'target_critic_1': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_critic_1.pth',
                    'target_critic_2': r'D:\PythonProject\Drone_self_avoidance\Self_Avoidance\target_critic_2.pth'}
    # 得到返回的奖励列表
    return_list = tools.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, pth_load, retrain,
                                               train_model, actor_pth_load, bn_txt, device)

    # 绘图
    episodes_list = list(range(len(return_list)))
    mv_return = tools.moving_average(return_list, 9)
    # 新建一个二维图形窗口
    fig_2d = plt.figure()
    # 添加一个二维子图
    ax_2d = fig_2d.add_subplot(1, 1, 1)
    ax_2d.plot(episodes_list, return_list, color='blue', label='原本训练记录')
    ax_2d.plot(episodes_list, mv_return, color='orange', label='滑动平均记录')

    ax_2d.legend()
    ax_2d.set_xlabel('轮次')
    ax_2d.set_ylabel('返回值')
    ax_2d.set_title('综合')
    # 显示图形
    plt.show()