"""
该文件用于编写一些常用的工具函数或者类型
"""
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import time
import os


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity):
        # 一个先进先出的队列
        # 创建了一个具有固定容量（capacity）的双端队列（deque）
        # maxlen 参数指定了经验缓冲区的最大容量，即队列中元素的最大数量。
        # 当缓冲区达到最大容量时，添加新元素会导致旧元素从队列的另一端被移除，以保持缓冲区不超过设定的最大值。
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
       将数据加入buffer中
       :param state: 状态
       :param action: 动作
       :param reword: 奖励
       :param next_state:动作之后的下一状态
       :param done:环境是否结束
       :return: 更新容器
       """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从缓冲区挑选数量为batch_size的样本
        :param batch_size: 要挑选的数量
        :return: 状态，动作，奖励，下一状态，done
        """
        # 从经验回放缓冲区中无放回地随机选择batch_size个样本（但是选择后缓冲区中经验数量不变）
        transitions = random.sample(self.buffer, batch_size)
        # 将 transitions 中的元素按列解压缩成五个列表
        # 并分别赋值给 state、action、reward、next_state 和 done。
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """
        读取目前缓冲区中数据的数量
        :return: 缓冲区中数据数量
        """
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def level_up(success_list):
    """
    用于判断是否提升课程学习难度等级的函数，现在的设计是当最后的10次中，有8次成功，即可升级
    :param success_list: 用于存放是否通过的列表
    :return: 升级与否
    """
    # 切片得到最后10个的情况
    count_list = success_list[-10:]
    # 清空计数器
    count = 0
    for i in range(len(count_list)):
        if count_list[i] == 1:
            count += 1
    # 如果10次迭代中有8次完成要求，升级
    if count >= 8:
        return True
    else:
        return False


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


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, pth_load, retrain,
                           max_eps_episode, min_eps, action_bound, train_model, device):
    """
    用于离线策略的训练函数
    :param env:可以交互的环境实例
    :param agent:智能体（含优化策略）
    :param num_episodes:迭代次数
    :param replay_buffer:经验回放池深度
    :param minimal_size:经验回放池最小深度
    :param batch_size:训练批次的经验
    :param pth_load:pth文件的存放地址字典
    :param retrain: 是否重新开始训练
    :param max_eps_episode: 最大贪心次数
    :param min_eps:最小贪心概率
    :param action_bound:动作限制值
    :param train_model:训练的模型是什么
    :param device: 设备
    :return:训练的结果
    """
    """初始化一些结果列表"""
    # 返回的结果，里面是每一次迭代得到的回报
    return_list = []
    # 记录每一次迭代，成功的无人机数量是否占到总数量的80%，用于判断等级是否提升
    success_list = []
    """加载存档点"""
    if not retrain:
        for name, pth in pth_load.items():
            # 按照键名称取出存档点
            check_point = torch.load(pth_load[name], map_location=device)
            # 装载模型参数
            agent.net_dict[name].load_state_dict(check_point['model'])
            # 装载优化器
            if name == 'actor' or name == 'critic':
                agent.net_optim[name].load_state_dict(check_point['optimizer'])
            # 取出训练的迭代次数，用于继续训练
            epoch_all = check_point['epoch']
    else:
        # 注意这里，如果权重不存在，需要先给迭代次数置0
        epoch_all = 0
    # 打印模型的参数
    """迭代训练过程"""
    for j in range(10):
        # 找到nan出现的地方
        # with torch.autograd.detect_anomaly():
        # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % j) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # 迭代的奖励总和
                episode_return = 0
                success_count = 0  # 统计成功到达目标区域的无人机数量
                crash_count = 0  # 坠毁无人机数量
                bt_count = 0  # 电量耗尽数量
                over_count = 0  # 超过最大步长的无人机
                epoch_all += 1  # 更新迭代总数
                n_done = 0  # 达到终止状态的无人机数量（包括成功到达目标、坠毁、电量耗尽等）
                # 得到状态的初始值，类型是np.ndarray
                # 环境重置 state是以无人机数目为行数，140为列数的列表
                state = env.reset()
                while True:
                    # 对于每个无人机对象来说
                    for i in range(len(env.uavs)):
                        if env.uavs[i].done:
                            # 无人机已结束任务，跳过
                            continue
                        # 更新步数，用于贪心策略的计算
                        agent.step = j * num_episodes / 10 + i_episode
                        # 将网络设置为验证模式，这样可以在BN层使用训练中得到的weight和bias
                        agent.actor.eval()
                        state_input = state[i]
                        state_input = torch.tensor(state_input, dtype=torch.float).to(device)
                        # 增加一个维度
                        state_input = torch.unsqueeze(state_input, dim=0)
                        # 选择动作，类型为np.ndarray
                        action = agent.take_action(state_input)[0]
                        # 得到下一个状态，奖励，是否完成，状态的类别（电量耗尽，坠机，到达目标点位等等）
                        next_state, reward, uav_done, info = env.step(action, i)  # 根据选取的动作改变状态，获取收益
                        env.uavs[i].info = info
                        # 将环境给出的奖励放到无人机对象的奖励记录中，用于检查每一步的好坏
                        env.uavs[i].reward.append(reward)
                        env.uavs[i].action.append(action)
                        episode_return += reward  # 求总收益
                        # 将状态、动作、奖励、下一状态、是否结束 打包放入经验回放池
                        replay_buffer.add(state[i], action, reward, next_state, uav_done)
                        """判断状态的类别"""
                        if info == 1:
                            # 如果到达目标点，完成目标的无人机数量加1
                            success_count += 1
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
                        # 如果达到经验回放池的最低要求
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        # 网络更新
                        agent.update(transition_dict)
                    # 如果一个批次的无人机全都训练完毕
                    if n_done == env.num_uavs:
                        break
                # 打印每一个无人机的奖励，看看
                for uav in env.uavs:
                    # print("reward:{}".format(uav.reward))
                    #     print("r_climb:{}".format(uav.r_climb))
                    #     print("r_target:{}".format(uav.r_target))
                    #     print("r_e:{}".format(uav.r_e))
                    #     print("c_p_crash:{}".format(uav.c_p_crash))
                    #     print("action:{}".format(uav.action))
                    #     print("r_n_distance:{}".format(uav.r_n_distance))
                    # 不看撞毁的
                    if uav.info == 2:
                        continue
                    print("state:{}".format(uav.now_state))
                    print("action:{}".format(uav.action))
                    #     print("r_n_distance:{}".format(uav.r_n_distance))
                # 如果有80%的无人机到达了目标区域
                if success_count >= 0.8 * env.num_uavs and env.level < 10:
                    # 记为一次效果较好的迭代
                    success_list.append(1)
                else:
                    success_list.append(0)
                # 判断长度大于10
                if len(success_list) >= 10:
                    # 得到升级与否
                    level_up_or_not = level_up(success_list)
                    # 如果可以升级同时环境的等级小于10，升级
                    if level_up_or_not and env.level < 10:
                        env.level += 1
                        success_list = []
                return_list.append(episode_return)
                # 打印相关训练信息
                print(
                    '\n 总迭代数量: {:5} 迭代次数: {:5} 得分: {:5}, 难度等级:{:5}  成功数量:{:2}  撞毁数量:{:2}  能量耗尽数量:{:2}  超过步长数量:{:2}'. \
                        format(epoch_all, (j * num_episodes / 10 + i_episode + 1), episode_return, env.level,
                               success_count,
                               crash_count, bt_count, over_count))
                """学习率更新"""
                # 是按照总体的迭代次数来更新的
                if train_model == 'DDPG':
                    agent.actor_scheduler.step()
                    agent.critic_scheduler.step()
                if train_model == 'SAC':
                    agent.actor_scheduler.step()
                    agent.critic_1_scheduler.step()
                    agent.critic_2_scheduler.step()
                """保存模型参数"""
                if (i_episode + 1) % 10 == 0:
                    # 保存批量状态，用于验证
                    # 每100周期保存一次网络参数
                    if train_model == 'DDPG':
                        state = {'model': agent.actor.state_dict(), 'optimizer': agent.actor_optimizer.state_dict(),
                                 'epoch': epoch_all}
                        torch.save(state, pth_load['actor'])
                        state = {'model': agent.critic.state_dict(), 'optimizer': agent.critic_optimizer.state_dict(),
                                 'epoch': epoch_all}
                        torch.save(state, pth_load['critic'])
                        state = {'model': agent.target_actor.state_dict(), 'epoch': epoch_all}
                        torch.save(state, pth_load['target_actor'])
                        state = {'model': agent.target_critic.state_dict(), 'epoch': epoch_all}
                        torch.save(state, pth_load['target_critic'])
                    if train_model == 'SAC':
                        state = {'model': agent.actor.state_dict(), 'optimizer': agent.actor_optimizer.state_dict(),
                                 'epoch': epoch_all}
                        torch.save(state, pth_load['SAC_actor'])
                        state = {'model': agent.critic_1.state_dict(),
                                 'optimizer': agent.critic_1_optimizer.state_dict(),
                                 'epoch': epoch_all}
                        torch.save(state, pth_load['critic_1'])
                        state = {'model': agent.critic_2.state_dict(),
                                 'optimizer': agent.critic_2_optimizer.state_dict(),
                                 'epoch': epoch_all}
                        torch.save(state, pth_load['critic_2'])
                        state = {'model': agent.target_critic_1.state_dict(), 'epoch': epoch_all}
                        torch.save(state, pth_load['target_critic_1'])
                        state = {'model': agent.target_critic_2.state_dict(), 'epoch': epoch_all}
                        torch.save(state, pth_load['target_critic_2'])
                # 绘制进度条相关的代码
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * j + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


if __name__ == '__main__':
    pass
