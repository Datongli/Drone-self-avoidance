"""
工具函数文件
"""
import collections
import numpy as np
import random


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity: int) -> None:
        # 一个先进先出的队列
        # 创建了一个具有固定容量（capacity）的双端队列（deque）
        # maxlen 参数指定了经验缓冲区的最大容量，即队列中元素的最大数量。
        # 当缓冲区达到最大容量时，添加新元素会导致旧元素从队列的另一端被移除，以保持缓冲区不超过设定的最大值。
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: dict, action: np.array, reward: float, nextState: dict, done: bool) -> tuple:
        """
        将数据加入buffer中
        :param state: 状态
        :param action: 动作
        :param reword: 奖励
        :param next_state:动作之后的下一状态
        :param done:环境是否结束
        :return: 更新容器
        """
        self.buffer.append((state.copy(), action.copy(), reward, nextState.copy(), done))

    def sample(self, batchSize: int) -> tuple:
        """
        从缓冲区挑选数量为batchSize的样本
        :param batchSize: 要挑选的数量
        :return: 状态，动作，奖励，下一状态，done
        """
        # 从经验回放缓冲区中无放回地随机选择batch_size个样本（但是选择后缓冲区中经验数量不变）
        transitions = random.sample(self.buffer, batchSize)
        # 将 transitions 中的元素按列解压缩成五个列表
        # 并分别赋值给 state、action、reward、next_state 和 done。
        state, action, reward, nextState, done = zip(*transitions)
        return np.array(state), action, reward, np.array(nextState), done

    def size(self) -> int:
        """
        读取目前缓冲区中数据的数量
        :return: 缓冲区中数据数量
        """
        return len(self.buffer)
    

if __name__ == '__main__':
    pass