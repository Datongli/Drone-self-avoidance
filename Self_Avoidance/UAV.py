"""
该文件用于编写无人机的类
"""
import numpy as np
import random
import math


class UAV:
    """
    无人机的类
    """
    def __init__(self, x, y, z, agent_r, env):
        """
        初始化无人机类
        :param x:无人机x初始坐标
        :param y:无人机y初始坐标
        :param z:无人机z初始坐标
        :param env:环境的类
        """
        # 初始化无人机位置
        self.x = x
        self.y = y
        self.z = z
        # 得到无人机的半径大小
        self.agent_r = agent_r
        # 得到环境类
        self.env = env
        # 初始化无人机运动情况
        self.bt = 5000  # 无人机电量
        self.p_bt = 10  # 无人机基础能耗，能耗/步
        self.k_bt = 2  # 无人机能耗系数，暂定为2
        self.now_bt = 4  # 无人机当前状态能耗
        self.cost = 0  # 无人机已经消耗能量
        self.step = 0  # 无人机已走步数
        self.p_crash = 0  # 无人机坠毁概率
        self.done = False  # 终止状态
        self.nearest_distance = 5  # 最近的障碍物的距离
        self.obstacle_num = 0
        # 无人机初始状态距离目标点的距离
        self.d_origin = math.sqrt((self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 无人机当前距离目标点的距离
        self.distance = math.sqrt((self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 无人机的感受野，注意，为了节省算力，无人机的可视范围定为了2，长宽高方向上都是2，2+1+2=5 5^3-1=124
        # 124个点的解释：5层，每层25个点，去掉中间的一个（无人机占据）
        self.ob_space = np.zeros(124)  # 无人机邻近栅格障碍物情况
        # 无人机从环境中得到的奖励
        self.reward = []
        # 一些想要被观测到的值
        self.info = 0
        self.r_climb = []
        self.r_target = []
        self.r_e = []
        self.c_p_crash = []
        self.action = []
        self.r_n_distance = []
        self.now_state = []

    def check_collision(self, sphere_center, sphere_radius, min_bound, max_bound):
        """
        判断球体和立方体是否相撞，以及计算距离的函数
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
        # 返回是否相撞、距离
        return distance <= sphere_radius, distance

    def col_limit_distance(self, sphere_center):
        """
        判断无人机对象和六个边界的最近距离
        :param sphere_center: 无人机的中心点，array类型的一维数组
        :return: 无人机和边界的最近距离
        """
        # 用于承接距离的列表
        distance = []
        # xyz三维坐标
        for i in range(3):
            # 每个方向上两个边界
            for j in range(2):
                # 直接做减法即可
                distance.append(abs(sphere_center[i] - self.env.action_area[j][i]))
        # 返回最小距离
        return min(distance)

    def state(self):
        """
        用于得到无人机状态的函数
        :return: 无人机的状态
        """
        # 得到三个坐标的差值
        dx = self.env.target.x - self.x
        dy = self.env.target.y - self.y
        dz = self.env.target.z - self.z
        # 状态网格
        state_grid = [self.x, self.y, self.z, dx, dy, dz,
                      self.env.target.x, self.env.target.y, self.env.target.z,
                      self.d_origin, self.step, self.distance,
                      self.p_crash, self.now_bt, self.cost]
        # 更新邻近栅格状态
        self.ob_space = []
        # 传感器点位是否有障碍
        self.obstacle_num = 0
        # 根据无人机周围的环境，更新无人机感受野中的情况
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # 将无人机原点扣掉
                    if i == 1 and j == 0 and k == 0:
                        continue
                    # 初始化最短距离
                    nearest_distance = 5
                    # 判断与建筑物的最近距离
                    for building in self.env.bds:
                        # 计算距离建筑物的最近距离
                        obstacle, build_distance = self.check_collision(np.array([self.x + i, self.y + j, self.z + k]), self.agent_r, building.left_down, building.right_up)
                        if build_distance <= nearest_distance:
                            # 更新最近距离
                            nearest_distance = build_distance
                    # 计算与边界的最近距离
                    limit_distance = self.col_limit_distance(np.array([self.x + i, self.y + j, self.z + k]))
                    if limit_distance <= nearest_distance:
                        nearest_distance = limit_distance
                    state_grid.append(nearest_distance)
        # 得到无人机的状态
        # 728+16=744
        return state_grid

    def update(self, action):
        """
        无人机状态等更新函数
        :param action:无人机选择的动作
        :return:奖励，是否结束一轮迭代，额外信息
        """
        # 个人认为每次应该给它赋初值，不然就是上一个状态的最近距离
        self.nearest_distance = 10
        """相关参数"""
        b = 3  # 撞毁参数 原3
        # wt = 0.05  # 目标参数0.005
        wx = 0.07
        wy = 0.07
        wz = 0.07  # 爬升参数 原0.07
        we = 0.2  # 能量损耗参数  原0.2 ，0
        crash = 3  # 坠毁概率惩罚增益倍数 原3 ，0
        """计算无人机坐标变更值"""
        dx = action[0]
        dy = action[1]
        dz = action[2]
        # 如果无人机静止不动，给予大量惩罚，但是可以继续运行
        # if math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) <= 0.1:
        #     return -1000, False, False
        """更新无人机的坐标值"""
        self.x += dx
        self.y += dy
        self.z += dz
        # 距离变化量，正代表接近目标，负代表远离目标
        Ddistance = self.distance - math.sqrt((self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 更新距离值
        self.distance = math.sqrt((self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 更新无人机走过的总步数
        self.step += 1
        """计算能耗与相关奖励"""
        # 更新能量损耗状态
        self.cost += self.now_bt
        # 当前能耗=基础能耗+能耗系数*距离变化量
        self.now_bt = self.p_bt + self.k_bt * (abs(dx) + abs(dy) + abs(dz))
        # 能耗与相关奖励  能耗系数*当前能耗
        r_e = - we * self.now_bt
        """计算碰撞概率与相应奖励"""
        # 无人机和最近建筑物是否碰撞
        collision_or_not = []
        """计算与障碍物之间的最短距离"""
        for building in self.env.bds:
            # 判断是否相撞同时求得无人机与建筑物之间的距离
            collision, build_distance = self.check_collision(np.array([self.x, self.y, self.z]), self.agent_r, building.left_down, building.right_up)
            # 得到是否与每一个建筑物相撞
            collision_or_not.append(collision)
            # 如果距离小于了最短距离
            if build_distance <= self.nearest_distance:
                # 更新最短距离
                self.nearest_distance = build_distance
        """计算与六个边界的距离，同时更新最短距离"""
        # 得到与六个边界的最近距离
        limit_distance = self.col_limit_distance(np.array([self.x, self.y, self.z]))
        if limit_distance <= self.nearest_distance:
            # 更新最近距离
            self.nearest_distance = limit_distance
        """计算坠毁概率p_crash"""
        # 如果最近距离大于5
        if self.nearest_distance >= 5:
            # 撞毁概率等于0
            self.p_crash = 0
        else:
            # 根据公式计算撞毁概率
            # b:撞毁参数；增加一个1e-5是为了防止出现除以0的现象
            self.p_crash = math.exp(b * 1 / (self.nearest_distance + 0.1))
        # 计算爬升奖励    wc：爬升系数
        r_climb = -wx * abs(self.x - self.env.target.x) - wy * abs(self.y - self.env.target.y) - wz * abs(self.z - self.env.target.z)
        # 最小距离奖励
        if self.nearest_distance >= 5:
            r_n_distance = 0
        else:
            # 让这个奖励绝对值大于1
            r_n_distance = -1 / (self.nearest_distance + 1e-5) * 10
        # 计算目标奖励，感觉有可能是目标奖励不明显，如果动作量接近的非常小的话
        # 感觉应该更改一下目标奖励的计算方式
        if self.distance > 1:
            r_target = 2 * (self.d_origin / self.distance) * Ddistance
        # 如果距离太近，已经是1了
        else:
            r_target = 2 * self.d_origin
        """计算总奖励r"""
        # 爬升奖励+目标奖励+能耗奖励-坠毁系数*坠毁概率
        reword = r_climb + r_target + r_e - crash * self.p_crash + r_n_distance
        # reword = r_climb + r_target + r_e - crash * self.p_crash
        self.r_climb.append(r_climb)
        self.r_target.append(r_target)
        self.r_e.append(r_e)
        self.c_p_crash.append(- crash * self.p_crash)
        self.r_n_distance.append(r_n_distance)
        self.now_state.append([self.x, self.y, self.z])
        # print(self.p_crash)
        # print("reword:{}".format(reword))
        """终止状态判断"""
        if (self.x <= 1 or self.x >= self.env.action_area[1][0] - 1
                or self.y <= 1 or self.y >= self.env.action_area[1][1] - 1
                or self.z <= 1 or self.z >= self.env.action_area[1][2] - 1
                or True in collision_or_not):
            # 发生碰撞，产生巨大惩罚
            # 根据碰撞点距离目标的远近来设置奖励的大小
            if self.distance <= 10:
                return reword - 2000, True, 2
            else:
                return reword - 50000, True, 2
        if self.distance <= 4:
            # 到达目标点，给予大量奖励
            return reword + 50000, True, 1
        # 注意平衡探索步长和碰撞的关系
        if self.step >= 4 * self.d_origin + 4 * self.env.action_area[1][2]:
            # 步数超过最差步长（2*初始距离+2*空间高度），给予惩罚
            return reword - 100, True, 5
        if self.cost > self.bt:
            # 电量耗尽，给予大量惩罚
            return reword - 100, True, 3
        # 如果没有达到上述的终止状态，则继续运行
        return reword, False, 4


if __name__ == '__main__':
    pass