"""
此文件用于构建可以交互的环境
"""
import torch
import numpy as np


def calculate_distance(point1, point2):
    """
    计算两个点之间的三维欧几里得距离
    :param point1: 第一个点的坐标 (x1, y1, z1)
    :param point2: 第二个点的坐标 (x2, y2, z2)
    :return: 两点之间的距离
    """
    distance = np.sqrt(np.sum((np.array(point2) - np.array(point1))**2))
    return distance


def check_collision(sphere_center, sphere_radius, min_bound, max_bound):
    """
    判断球体和立方体是否相撞
    :param sphere_center:球体球心
    :param sphere_radius: 球体半径
    :param min_bound:立方体最左下角的点
    :param max_bound:立方体最右上角的点
    :return:是否相撞，距离
    """
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)
    # 将球心坐标限制在立方体边界内
    closest_point = np.clip(sphere_center, min_bound, max_bound)
    # 计算球心到最近点的距离
    distance = np.linalg.norm(sphere_center - closest_point)
    # 判断球体是否在立方体内或相交
    return distance <= sphere_radius, distance


class Environment:
    """
    用于和智能体交互的环境实例
    """
    def __init__(self, obstacle, agent_state, agent_r, target_area, action_area):
        # 障碍物字典
        self.obstacle = obstacle
        # 智能体的初始化位置(不变)
        self.state_initial = agent_state
        # 智能体的位置(可改变)
        self.state = self.state_initial
        # 智能体的半径
        self.agent_r = agent_r
        # 动作的步数
        self.steps = 0
        # 得到目标区域（对角线形式）
        self.target_area = target_area
        # 环境给的奖励值
        self.reward = 0
        # 可以运动的空间(对角线形式)
        self.action_area = action_area

    def reset(self):
        """
        重置智能体
        :return: 智能体的初始位置
        """
        # 步数清零
        self.steps = 0
        # 智能体位置归位
        self.state = self.state_initial
        # 奖励值清零
        self.reward = 0
        return self.state

    def collision(self, state):
        """
        判断智能体是否与障碍物相撞或者边界相撞的函数
        :param state: 智能体当前的状态
        :return: 相撞与否的状态
        """
        # 初始化“距离”字典
        distance_dist = {"sphere": [],  # 球体
                         "cube": [],  # 立方体
                        "boundary": []  # 边界
                         }
        # 初始化“是否相撞”的指标
        collision_or_not = []
        # 得到球体（半球）障碍的个数
        sphere_num = len(self.obstacle['sphere'])
        for i in range(sphere_num):
            # 可能需要先将state转化为列表类型的数据
            # 得到球心的坐标
            spherical_center = self.obstacle['sphere'][i][0]
            # 得到半径
            spherical_r = self.obstacle['sphere'][i][1]
            # 计算智能体和球形目标之间的距离
            distance = calculate_distance(spherical_center, state)
            distance_dist['sphere'].append(distance)
            # 如果小于两个球体的半径和，判断为相撞
            if distance <= self.agent_r + spherical_r:
                collision_or_not.append(True)
            else:
                collision_or_not.append(False)
        # 得到立方体障碍
        cub = self.obstacle['cube']
        # 得到立方体障碍的个数
        cube_num = len(cub)
        for i in range(cube_num):
            # 判断智能体是否与立方体障碍相撞，并得到距离
            pd, distance = check_collision(state, self.agent_r, cub[i][0], cub[i][1])
            distance_dist['cube'].append(distance)
            if pd:
                collision_or_not.append(True)
            else:
                collision_or_not.append(False)
        # 判断是否与边界发生碰撞
        for i in range(len(state)):
            if self.agent_r <= state[i] <= (100 - self.agent_r):
                collision_or_not.append(False)
            else:
                collision_or_not.append(True)
        # 计算与边界的距离
        for i in range(len(self.action_area)):
            for j in range(len(self.action_area[i])):
                distance_dist['boundary'].append(self.state[j] - self.action_area[i][j])
        # 返回是否碰撞和距离字典
        if True in collision_or_not:
            return True, distance_dist
        else:
            return False, distance_dist

    def step(self, action):
        """
        环境更新函数
        :param action: 动作量，尽量用np.array
        :return: 下一个状态，奖励，是否结束
        """
        # 步数更新
        self.steps += 1
        # 奖励值更新
        self.reward += -1 * 1
        # 结束与否
        done = False
        # 判断是否有碰撞发生，同时计算障碍物和智能体之间的距离
        col_or_not, obstacle_distance_dist = self.collision(self.state)
        if col_or_not:
            # 如果相撞，奖励-100，结束
            self.reward += -100
            done = True
        else:
            # 如果没有相撞
            for i in range(len(obstacle_distance_dist['sphere'])):
                self.reward += -10 * (1 / obstacle_distance_dist['sphere'][i])
            for i in range(len(obstacle_distance_dist['cube'])):
                self.reward += -10 * (1 / obstacle_distance_dist['cube'][i])
            for i in range(len(obstacle_distance_dist['boundary'])):
                self.reward += -10 * (1 / abs(obstacle_distance_dist['boundary'][i]))
        # 判断是否到达目标区域
        arrive_or_not, distance = check_collision(self.state, self.agent_r, self.target_area[0], self.target_area[1])
        if arrive_or_not:
            # 如果到达，奖励200，结束
            self.reward += 200
            done = True
        else:
            # 如果没有相撞
            self.reward += -10 * distance
        # 得到下一个状态量（这一版是直接在xyz坐标上加减）
        next_state = self.state + action
        # 更新状态
        self.state = next_state
        return next_state, self.reward, done









if __name__ == "__main__":
    pass


