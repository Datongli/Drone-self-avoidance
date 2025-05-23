"""
此文件用于构建可以交互的环境
"""
import torch
import numpy as np
import math
import random
import UAV
import matplotlib.pyplot as plt


class Building:
    """
    建筑物的类
    """

    def __init__(self, x, y, length, width, height, left_down, right_up):
        self.x = x  # 建筑物中心的x坐标
        self.y = y  # 建筑物中心的y坐标
        self.length = length  # 建筑物x方向长度一半
        self.width = width  # 建筑物y方向宽度一半
        self.height = height  # 建筑物高度
        self.left_down = left_down  # 建筑物左下角点的坐标
        self.right_up = right_up  # 建筑物右上角点的坐标


class Target:
    """
    目标点的类别
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Environment:
    """
    用于和智能体交互的环境实例
    """

    def __init__(self, agent_r, action_area, num_nuvs, v0):
        # 无人机的状态
        self.state = []
        # 无人机的半径
        self.agent_r = agent_r
        # 目标点位
        self.target = [0, 0, 0]
        # 可以运动的空间(对角线形式)
        self.action_area = action_area
        # 课程学习的水平(训练难度等级)
        self.level = 1
        # 无人机对象的集合，为了提升效率，每次不单单使用一个无人机进行搜索，可以是多个
        self.uavs = []
        # 建筑集合
        self.bds = []
        # 训练环境中的无人机个数
        self.num_uavs = num_nuvs
        # 无人机可控风速
        self.v0 = v0
        self.fig = plt.figure()  # 创建一个新的图形窗口并存储在self.fig中
        # 在图形窗口添加一个3D投影子图
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

    def reset(self):
        """
        重置环境
        将重新按照课程学习的水平，生成建筑物
        重新给无人机分配起始点，建立新的目标区域等等
        :return:重置后的状态
        """
        """清空无人机对象和建筑物对象"""
        # 清空无人机对象集合
        self.uavs = []
        # 清空建筑对象集合
        self.bds = []
        """构建建筑物"""
        # 随机生成建筑物，根据难度等级随机循环，同时判断是否有重叠
        # 要生成建筑物的数量，依据课程学习的难度进行
        building_num = random.randint(self.level, self.level * 2)
        # building_num = 1
        while True:
            # 判断是否达到要生成的数量
            if len(self.bds) >= building_num:
                break
            # 建筑物中心的x坐标
            x = random.uniform(20, self.action_area[1][0] - 10)
            # 建筑物中心的y坐标
            y = random.uniform(10, self.action_area[1][1] - 10)
            # 建筑物x方向长度的一半
            length = random.uniform(1, 5)
            # 建筑物y方向宽度的一半
            width = random.uniform(1, 5)
            # 建筑物高度
            height = random.uniform(self.action_area[1][2] - 20, self.action_area[1][2] - 3)
            # x = 50
            # y = 50
            # length = 30
            # width = 30
            # height = 20
            # 建筑物左下角的点
            left_down = [x - length, y - width, 0]
            # 建筑物右上角的点
            right_up = [x + length, y + width, height]
            """判断生成的建筑物是否有重叠现象"""
            if len(self.bds) == 0:
                # 如果是第一次生成，直接跳过判断，增加列表中元素
                self.bds.append(Building(x, y, length, width, height, left_down, right_up))
            else:
                # 是否重叠的标志位
                overlop_num = 0
                for building in self.bds:
                    # 如果没有重叠现象
                    if abs(x - building.x) >= length + building.length or abs(y - building.y) >= width + building.width:
                        continue
                    else:
                        overlop_num = 1
                        # 只要判断到有重叠，就不需要判断再判断了
                        break
                # 如果没有重叠
                if overlop_num == 0:
                    self.bds.append(Building(x, y, length, width, height, left_down, right_up))
        """随机生成目标点的位置"""
        while True:
            # 生成目标点位
            x = random.randint(60, 90)
            y = random.randint(10, 90)
            z = random.randint(10, self.action_area[1][2] - 5)
            # x = 90
            # y = 90
            # z = 20
            # 判断目标是否在障碍物中
            in_build = 0  # 标志位
            for building in self.bds:
                if (building.left_down[0] - 1 <= x <= building.right_up[0] + 1
                        and building.left_down[1] - 1 <= y <= building.right_up[1] + 1
                        and building.left_down[2] <= z <= building.right_up[2] + 1):
                    # 目标在障碍物中
                    in_build = 1
                    break
            if in_build == 0:
                self.target = Target(x, y, z)
                break
        """随机生成无人机的初始位置"""
        for uav_num in range(self.num_uavs):
            too_many_mun = 0  # 为了防止陷入死循环，标志位
            while True:
                # 生成初始坐标
                x = random.randint(10, 15)
                y = random.randint(10, 90)
                z = random.randint(6, 8)
                # x = 10
                # y = 50
                # z = 5
                # 判断是否与建筑物重合的标志位
                in_build = 0
                # 判断无人机和目标点连线上是否有障碍物的标志位
                # 在等级4后应该有，以增加环境的复杂性
                complex_flag = 0
                if self.level <= 0:
                    # 环境难度等级小于3的时候，没有必要这么复杂
                    complex_flag = 1
                # 确保没有生成在障碍物的区域
                for building in self.bds:
                    if (building.left_down[0] - 2 <= x <= building.right_up[0] + 2
                            and building.left_down[1] - 2 <= y <= building.right_up[1] + 2
                            and building.left_down[2] <= z <= building.right_up[2] + 2):
                        in_build = 1
                        break
                if in_build == 0:
                    # 判断无人机和目标点连线上是否有障碍物
                    uav_local = np.array([x, y, z])
                    tar_local = np.array([self.target.x, self.target.y, self.target.z])
                    # 计算两个点之间的向量
                    vector = tar_local - uav_local
                    # 计算两个点之间的距离
                    distance = np.linalg.norm(vector)
                    # 将向量标准化为单位向量
                    normed_vector = vector / distance
                    # 取100个点，离散化目标点和无人机之间的距离
                    mini_distance = distance / 100.0
                    for i in range(100):
                        if i == 98:
                            too_many_mun += 1
                        if too_many_mun >= 1000:
                            # 防止陷入死循环
                            complex_flag = 1
                        vector_pd = normed_vector * (i * mini_distance) + uav_local
                        for building in self.bds:
                            if (building.left_down[0] <= vector_pd[0] <= building.right_up[0]
                                    and building.left_down[1] <= vector_pd[1] <= building.right_up[1]
                                    and building.left_down[2] <= vector_pd[2] <= building.right_up[2]):
                                # 如果无人机和目标点之间有障碍物，则将complex_flag设为1
                                complex_flag = 1
                    if complex_flag == 1:
                        # 全部条件都满足，uav列表增加
                        uav = UAV.UAV(x, y, z, self.agent_r, self)
                        uav.num = uav_num + 1
                        self.uavs.append(uav)
                        break
        # 更新无人机状态 np.vstack: 按照行方向堆叠数组  uav.state()是长度为140的列表，代表了各种状态
        self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])
        return self.state

    def step(self, action, i):
        """
        环境更新函数
        :param action: 无人机提供的动作array类型的数据
        :param i:第i个无人机
        :return: 下一个状态，奖励，是否结束，用于调试的额外信息等
        """
        # 无人机执行行为,info为是否到达目标点
        reward, done, info = self.uavs[i].update(action[0])
        next_state = self.uavs[i].state()
        return next_state, reward, done, info

    def render(self, flag=0):
        """
        绘制封闭的立方体的函数，用于查验模型的性能
        :param flag:是否是第一次渲染
        :return:绘图
        """
        if flag == 1:
            for building in self.bds:
                # 绘画出所有建筑
                x = building.x
                y = building.y
                z = 0
                dx = building.length
                dy = building.width
                dz = building.height
                xx = np.linspace(x - dx, x + dx, 2)
                yy = np.linspace(y - dy, y + dy, 2)
                zz = np.linspace(z, z + dz, 2)
                xx2, yy2 = np.meshgrid(xx, yy)
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z + dz))
                yy2, zz2 = np.meshgrid(yy, zz)
                self.ax.plot_surface(np.full_like(yy2, x - dx), yy2, zz2)
                self.ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2)
                xx2, zz2 = np.meshgrid(xx, zz)
                self.ax.plot_surface(xx2, np.full_like(yy2, y - dy), zz2)
                self.ax.plot_surface(xx2, np.full_like(yy2, y + dy), zz2)
                sn = self.target
                # 绘制目标坐标点
                self.ax.scatter(sn.x, sn.y, sn.z, c='red')
        for uav in self.uavs:
            # 绘制无人机坐标点
            self.ax.scatter(uav.x, uav.y, uav.z, c='blue')



if __name__ == "__main__":
    pass
