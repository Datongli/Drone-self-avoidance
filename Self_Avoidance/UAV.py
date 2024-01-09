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
        self.dir = 0  # 无人机水平运动方向(弧度)
        self.p_bt = 10  # 无人机基础能耗，能耗/步
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
        if distance <= sphere_radius:
            dir_ob = 0
        else:
            # 计算球心和障碍物在水平面上投影的夹角  同时保证在（-1， 1）范围内，使反三角函数不会定义错误
            if (sphere_center[0] - closest_point[0]) ** 2 + (sphere_center[1] - closest_point[1]) ** 2 == 0:
                cosine_value = 1
            else:
                cosine_value = abs(sphere_center[0] - closest_point[0]) / ((sphere_center[0] - closest_point[0]) ** 2 + (sphere_center[1] - closest_point[1]) ** 2)
            cosine_value = max(-1, min(1, cosine_value))
            # 使用 math.acos
            dir_ob = math.acos(cosine_value)
        # 判断球体是否在立方体内或相交
        return distance <= sphere_radius, distance, dir_ob

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
        area_x = self.env.action_area[1][0]
        area_y = self.env.action_area[1][1]
        area_z = self.env.action_area[1][2]
        # 状态网格
        state_grid = [self.x, self.y, self.z, dx, dy, dz,
                      self.env.target.x, self.env.target.y, self.env.target.z,
                      self.d_origin, self.step, self.distance, self.dir,
                      self.p_crash, self.now_bt, self.cost]
        # state_grid = [self.x / area_x, self.y / area_y, self.z / area_z,
        #               dx / area_x, dy / area_y, dz / area_z,
        #               self.env.target.x / area_x, self.env.target.y / area_y, self.env.target.z / area_z,
        #               self.d_origin / math.sqrt(area_x ** 2 + area_y ** 2 + area_z ** 2),
        #               self.step / (2 * self.d_origin + 2 * area_z), self.distance / self.d_origin, self.dir,
        #               self.p_crash, self.now_bt, self.cost / self.now_bt]
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
                    nearest_distance = 5
                    # 判断与建筑物的最近距离
                    for building in self.env.bds:
                        # 计算距离建筑物的最近距离
                        _, build_distance, _ = self.check_collision(np.array([self.x + i, self.y + j, self.z + k]), self.agent_r, building.left_down, building.right_up)
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
        self.nearest_distance = 5
        # 更新无人机状态
        dx, dy, dz = [0, 0, 0]
        """相关参数"""
        b = 3  # 撞毁参数 原3
        # wt = 0.05  # 目标参数0.005
        wc = 0.1  # 爬升参数 原0.07
        wx = 0.1  # x轴的接近系数
        wy = 0.1  # y轴的接近系数
        we = 0.2  # 能量损耗参数  原0.2 ，0
        c = 0.05  # 风阻能耗参数 原0.05
        crash = 3  # 坠毁概率惩罚增益倍数 原3 ，0
        """计算无人机坐标变更值"""
        dx = action[0]
        dy = action[1]
        dz = action[2]
        # 如果无人机静止不动，给予大量惩罚
        if math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) <= 0.01:
            return -1000, False, False
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
        """考虑风场的影响"""
        # 一个标志位，判断无人机是否沿着y轴方向运动
        self.flag = 1
        if abs(dy) == dy:
            self.flag = 1
        else:
            self.flag = -1
        # 得到无人机速度方向（弧度），用于计算风场的影响
        if dx * dx + dy * dy != 0:
            self.dir = math.acos(min(1, max(-1, dx / math.sqrt(dx * dx + dy * dy)))) * self.flag
        """计算能耗与相关奖励"""
        # 更新能量损耗状态
        self.cost += self.now_bt
        # 无人机速度方向与风速方向夹角
        a = abs(self.dir - self.env.WindField[1])
        # 当前能耗=基础能耗+风阻能耗参数*风力*风向（投影）
        self.now_bt = self.p_bt + c * self.env.WindField[0] * (math.sin(a) - math.cos(a))
        # 能耗与相关奖励
        # r_e = we * (self.p_bt - self.now_bt)
        r_e = 0
        """计算碰撞概率与相应奖励"""
        # 无人机和最近建筑物的水平夹角
        dir_ob = 0
        collision_or_not = []
        """计算与障碍物之间的最短距离"""
        for building in self.env.bds:
            # 判断是否相撞同时求得无人机与建筑物之间的距离
            collision, build_distance, dir_bud = self.check_collision(np.array([self.x, self.y, self.z]), self.agent_r, building.left_down, building.right_up)
            # 得到是否与每一个建筑物相撞
            collision_or_not.append(collision)
            # 如果距离小于了最短距离
            if build_distance <= self.nearest_distance:
                # 更新最短距离
                self.nearest_distance = build_distance
                # 更新无人机与最近建筑物的水平夹角
                dir_ob = dir_bud
        """计算与六个边界的距离，同时更新最短距离"""
        # 得到与六个边界的最近距离
        limit_distance = self.col_limit_distance(np.array([self.x, self.y, self.z]))
        if limit_distance <= self.nearest_distance:
            # 更新最近距离
            self.nearest_distance = limit_distance
        """计算坠毁概率p_crash"""
        # 如果最近距离大于6同时风速小于可控风速
        if self.nearest_distance >= 5 and self.env.WindField[0] <= self.env.v0:
            # 撞毁概率等于0
            self.p_crash = 0
        else:
            # 根据公式计算撞毁概率
            # b:撞毁参数；v0:无人机可控风速
            self.p_crash = math.exp(-b * self.nearest_distance * self.env.v0 * self.env.v0 /
                                    (0.5 * math.pow(self.env.WindField[0] *
                                    math.cos(abs(self.env.WindField[1] - dir_ob) - self.env.v0), 2)))
        # 计算爬升奖励    wc：爬升系数
        r_climb = - wc * abs(self.z - self.env.target.z) - wy * abs(self.y - self.env.target.y) - wx * abs(self.x - self.env.target.x)
        # r_climb = 0
        # 最小距离奖励
        if self.nearest_distance >= 5:
            r_n_distance = 0
        else:
            r_n_distance = - self.obstacle_num / 42 * 10 - 1 / self.nearest_distance * 10
        # 计算目标奖励
        if self.distance > 1:
            r_target = 0.2 * (self.d_origin / self.distance) * Ddistance
        # 如果距离太近，已经是1了
        else:
            r_target = 0.2 * (self.d_origin) * Ddistance
        """计算总奖励r"""
        # 爬升奖励+目标奖励+能耗奖励-坠毁系数*坠毁概率
        # reword = r_climb + r_target + r_e - crash * self.p_crash + r_n_distance
        reword = r_climb + r_target + r_e - crash * self.p_crash
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
                or True in collision_or_not
                or random.random() < self.p_crash):
            # 发生碰撞，产生巨大惩罚
            return reword - 4000, True, 2
        if self.distance <= 4:
            # 到达目标点，给予大量奖励
            return reword + 4000, True, 1
        if self.step >= 2 * self.d_origin + 2 * self.env.action_area[1][2]:
            # 步数超过最差步长（2*初始距离+2*空间高度），给予惩罚
            return reword - 10, True, 5
        if self.cost > self.bt:
            # 电量耗尽，给予大量惩罚
            return reword - 20, True, 3
        # 如果没有达到上述的终止状态，则继续运行
        return reword, False, 4


if __name__ == '__main__':
    pass