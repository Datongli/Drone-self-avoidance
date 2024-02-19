######################################################################
# DQN Model Train
#---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
#对训练参数进行设置，并对基于DQN的无人机航迹规划算法模型进行训练

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from env import *
from collections import deque
from replay_buffer import ReplayMemory, Transition
import torch
import torch.optim as optim
import random
from model import QNetwork


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


use_cuda = torch.cuda.is_available()
# 根据cuda是否可用，选择CUDA的张量类型
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 128    # 批量大小
# TAU = 0.005
gamma = 0.99   # 折扣率
LEARNING_RATE = 0.0004   # 学习率
TARGET_UPDATE = 10   # Q网络更新周期

num_episodes = 40000  # 训练周期长度
print_every = 1  # 每训练print_every周期，打印一次结果
hidden_dim = 16  # 网络隐藏层的节点数
min_eps = 0.01    # 最小贪心概率
max_eps_episode = 10   # 最大贪心次数

space_dim = 42  # n_spaces   状态空间维度 26+16，在UAV里有解释
action_dim = 27  # n_actions   动作空间维度 这是一个离散的动作空间，相当于建立了一个三层的点，每层9个，一共27个
print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

threshold = 200  # 门槛值
env = Env(space_dim, action_dim, LEARNING_RATE, hidden_dim)
print('threshold: ', threshold)


def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    """
    伊布西龙衰减函数，用于计算贪心概率
    :param i_epsiode:当前的迭代次数
    :param max_episode:最大的贪心次数
    :param min_eps:最小的伊布西龙值，表示希望在训练结束时，模型趋向于完全贪心
    :return:贪心概率
    """
    # 计算斜率，用于线性衰减
    slope = (min_eps - 1.0) / max_episode
    # 其实在当前的斜率（-0.099）下，迭代10次后，就已经为min_eps了
    # 是一个先线性下降，再恒定的过程，最后恒定在0.01
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps        


def save(directory, filename):   # 存放Q网络参数
    torch.save(env.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(env.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))


def run_episode(env, eps):
    """
    运行一次迭代
    :param env:实例化的环境变量
    :param eps:贪心概率
    :return:
    """
    # 环境重置 state是以无人机数目为行数，42为列数的列表
    state = env.reset()
    total_reward = 0  # 每一次迭代的奖励总和
    n_done = 0  # 记录所有完成任务的无人机总数
    count = 0  # 记录迭代的步数
    success_count = 0  # 统计任务完成数量
    crash_count = 0  # 坠毁无人机数量
    bt_count = 0  # 电量耗尽数量
    over_count = 0  # 超过最大步长的无人机
    while True:
        count += 1  # 步数加1
        # 对于每个无人机对象来说
        for i in range(len(env.uavs)):
            if env.uavs[i].done:
                # 无人机已结束任务，跳过
                continue
            # 根据Q值选取动作
            action = env.get_action(FloatTensor(np.array([state[i]])), eps)
            # 得到下一个状态，奖励，是否完成，状态的类别（电量耗尽，坠机，到达目标点位等等）
            next_state, reward, uav_done, info = env.step(action.detach(), i)  # 根据选取的动作改变状态，获取收益
            total_reward += reward  # 求总收益
            # 将状态、动作、奖励、下一状态、是否结束 打包放入经验回放池
            env.replay_memory.push(
                    (FloatTensor(np.array([state[i]])), 
                    action, # action is already a tensor
                    FloatTensor([reward]), 
                    FloatTensor([next_state]), 
                    FloatTensor([uav_done])))
            """判断状态的类别"""
            if info == 1:
                # 如果到达目标点，完成目标的无人机数量加1
                success_count = success_count + 1
            elif info == 2:
                # 如果发生碰撞，发生碰撞的无人机数量加1
                crash_count += 1
            elif info == 3:
                # 如果电量耗尽，电量耗尽的无人机数量加1
                bt_count += 1
            elif info == 5:
                # 如果步数超过最差步数，超出的无人机数量加1
                over_count += 1
            """如果是结束状态"""
            if uav_done:
                # 更新无人机的状态
                env.uavs[i].done = True
                # 完成任务（包括到达目标，电量耗尽，发生碰撞，步长超过最差步等）
                n_done = n_done + 1
                continue
            # 状态变更
            state[i] = next_state
        #env.render()
        # 如果是训练步数5的整数倍，同时经验回放池的长度大于了batch_size（可以做训练）
        if count % 5 == 0 and len(env.replay_memory) > BATCH_SIZE:
            # batch = env.replay_memory.sample(BATCH_SIZE)
            env.learn(gamma, BATCH_SIZE)  # 训练Q网络
        # 如果无人机全部执行完毕
        if n_done >= env.n_uav:
            break
    if success_count >= 0.8 * env.n_uav and env.level < 10:
        # 通过率较大，难度升级
        env.level = env.level + 1
    return total_reward, [success_count, crash_count, bt_count, over_count]


def train():
    """
    训练过程的主函数
    :return: 分数数组和平均分数数组
    """
    # 建立一个双端队列，长度为100，用于存储得分
    scores_deque = deque(maxlen=100)
    # 得分列表
    scores_array = []
    # 平均得分列表
    avg_scores_array = []    
    
    # 得到时间
    time_start = time.time()
    # 载入存档点，通过这样的操作，可以从存档点继续训练，而不是每一次都从头开始训练
    check_point_Qlocal = torch.load('Qlocal.pth')
    check_point_Qtarget = torch.load('Qtarget.pth')
    # 取出模型参数
    env.q_target.load_state_dict(check_point_Qtarget['model'])
    env.q_local.load_state_dict(check_point_Qlocal['model'])
    # 取出优化器
    env.optim.load_state_dict(check_point_Qlocal['optimizer'])
    # 取出训练的迭代次数
    epoch = check_point_Qlocal['epoch']

    for i_episode in range(num_episodes):
        # 计算贪心概率
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        # 迭代一次，获得得分,返回到达目标的个数
        score, info = run_episode(env, eps)

        scores_deque.append(score)  # 添加得分
        scores_array.append(score)
        
        # 求双端队列里的平均得分
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        # 计算从time_start到当前时间经过了多长时间，并以秒数的整数存储在dt变量里面
        dt = (int)(time.time() - time_start)
            
        if i_episode % print_every == 0 and i_episode > 0:
            print('总迭代数量: {:5} 迭代次数: {:5} 得分: {:5}  平均得分: {:.2f}, 贪心概率: {:5.2f} 经历时间: {:02}:{:02}:{:02} 难度等级:{:5}  成功数量:{:2}  撞毁数量:{:2}  能量耗尽数量:{:2}  超过步长数量:{:2}'.\
                    format(i_episode+epoch, i_episode, score, avg_score, eps, dt//3600, dt%3600//60, dt%60, env.level, info[0], info[1], info[2], info[3]))
        # 保存模型参数
        if i_episode % 100 == 0:
            # 每100周期保存一次网络参数
            state = {'model': env.q_target.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode + epoch}
            torch.save(state, "Qtarget.pth")
            state = {'model': env.q_local.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode + epoch}
            torch.save(state, "Qlocal.pth")

        if i_episode % TARGET_UPDATE == 0:
            env.q_target.load_state_dict(env.q_local.state_dict()) 
    
    return scores_array, avg_scores_array

  


if __name__ == '__main__':
    # 开始训练，得到得分和平均得分
    scores, avg_scores = train()
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))
