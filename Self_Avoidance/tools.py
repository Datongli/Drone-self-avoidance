"""
该文件用于编写一些常用的工具函数或者类型
"""
from tqdm import tqdm
import numpy as np
import torch
import collections
import random


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


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """
    用于离线策略的训练函数
    :param env:可以交互的环境实例
    :param agent:智能体（含优化策略）
    :param num_episodes:迭代次数
    :param replay_buffer:经验回放池深度
    :param minimal_size:经验回放池最小深度
    :param batch_size:训练批次的经验
    :return:训练的结果
    """
    # 返回的结果，里面是每一次迭代得到的回报
    return_list = []
    for i in range(10):
        # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # 初始化回报计数
                episode_return = 0
                # 得到状态的初始值，类型是np.ndarray
                state = env.reset()
                done = False
                step = 0
                state_reword = []
                while not done:
                    # step += 1
                    # if step > 2000:
                    #     break
                    # 采取动作，类型为np.ndarry
                    action = agent.take_action(state)[0]
                    # 从环境实例中交互得到
                    # 下一状态，np.ndarray
                    # 奖励，np.float64
                    # 是否完成（达到目标或者失败）bool
                    next_state, reward, done = env.step(action)
                    # 将经验放入经验回放池
                    replay_buffer.add(state, action, reward, next_state, done)
                    # 更新状态
                    state = next_state
                    # 累加奖励
                    episode_return += reward
                    # 如果达到经验回放池的最低要求
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        # 更新智能体
                        agent.update(transition_dict)
                    state_reword.append(state)
                print(state_reword)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list