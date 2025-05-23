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
        self.p_bt = 5  # 无人机基础能耗，能耗/步
        self.k_bt = 0.2  # 无人机能耗系数，暂定为2
        self.now_bt = 4  # 无人机当前状态能耗
        self.cost = 0  # 无人机已经消耗能量
        self.step = 1  # 无人机已走步数
        self.p_crash = 0  # 无人机坠毁概率
        self.done = False  # 终止状态
        self.nearest_distance = 20  # 最近的障碍物的距离
        # 无人机初始状态距离目标点的距离
        self.d_origin = math.sqrt((self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 无人机当前距离目标点的距离
        self.distance = math.sqrt((self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 无人机的感受野，注意，为了节省算力，无人机的可视范围定为了2，长宽高方向上都是2，2+1+2=5 5^3-1=124
        # 124个点的解释：5层，每层25个点，去掉中间的一个（无人机占据）
        # 无人机从环境中得到的奖励
        self.reward = []
        # 无人机总奖励
        self.total_reward = 0
        # 一些想要被观测到的值
        self.info = 0
        self.r_climb = []
        self.r_target = []
        self.r_e = []
        self.c_p_crash = []
        self.action = []
        self.r_n_distance = []
        self.now_state = []
        self.num = 1

    def check_collision(self, sphere_center, sphere_radius, min_bound, max_bound, i=0, j=0, k=0):
        """
        判断球体和立方体是否相撞，以及计算距离的函数
        :param sphere_center:球体球心
        :param sphere_radius: 球体半径
        :param min_bound:立方体最左下角的点
        :param max_bound:立方体最右上角的点
        :param i: 无人机的x方向传感器点位
        :param j: 无人机的y方向传感器点位
        :param k: 无人机的z方向传感器点位
        :return:是否相撞，距离
        """
        zero_num = 0
        if i == 0:
            zero_num += 1
        if j == 0:
            zero_num += 1
        if k == 0:
            zero_num += 1
        min_bound = np.array(min_bound)
        max_bound = np.array(max_bound)
        # 将球心坐标限制在立方体边界内
        closest_point = np.clip(sphere_center, min_bound, max_bound)
        if zero_num == 2:
            ijk_same = 0
            pos_or_neg = 0
            # 如果ijk中有两个值为0，则证明这是上下前后左右里面的一个传感器
            for coordinate in range(3):
                # 让传感器只关注其正前方的障碍物，及传感器和障碍物的三维坐标中两个是相同的
                if sphere_center[coordinate] == closest_point[coordinate]:
                    ijk_same += 1
            if i != 0 and j == 0 and k == 0:
                # 左右传感器
                order_num = 0
                pos_or_neg = i
            if i == 0 and j != 0 and k == 0:
                # 前后传感器
                order_num = 1
                pos_or_neg = j
            if i == 0 and j == 0 and k != 0:
                # 上下传感器
                order_num = 2
                pos_or_neg = k
            if sphere_center[order_num] != closest_point[order_num]:
                if (pos_or_neg > 0 and sphere_center[order_num] < closest_point[order_num]) or (pos_or_neg < 0 and sphere_center[order_num] > closest_point[order_num]):
                    ijk_same += 0
                else:
                    ijk_same += 3
            else:
                ijk_same += 3
            if ijk_same == 2:
                distance = np.linalg.norm(sphere_center - closest_point)
            else:
                distance = 1000
        else:
            # 其他传感器直接计算球心到最近点的距离
            distance = np.linalg.norm(sphere_center - closest_point)
            # 在同一层的四个传感器，如果在距离内就直接置0，反之为1
            if distance <= sphere_radius + 0.5:
                # 现在假设传感器的探测范围是0.8，防止无人机运行一步直接进入障碍物
                distance = 0
            else:
                distance = 1000
        # 增加一个如果传感器已经在建筑物里的
        if np.array_equal(sphere_center, closest_point):
            distance = 0
        # 返回是否相撞、距离
        return distance <= sphere_radius, distance

    def col_limit_distance(self, sphere_center, i=0, j=0, k=0):
        """
        判断无人机对象和六个边界的最近距离
        :param sphere_center: 无人机的中心点，array类型的一维数组
        :param i: 无人机的x方向传感器点位
        :param j: 无人机的y方向传感器点位
        :param k: 无人机的z方向传感器点位
        :return: 无人机和边界的最近距离
        """
        # 用于承接距离的列表
        distance = []
        # xyz三维坐标
        for count1 in range(3):
            # 将几个特殊的传感器分离开
            if i == 0 and j == 0 and k != 0:
                count1 = 2
            if i == 0 and j != 0 and k == 0:
                count1 = 1
            if i != 0 and j == 0 and k == 0:
                count1 = 0
            # 每个方向上两个边界
            for count2 in range(2):
                if i == 0 and j == 0 and k == -1:
                    count2 = 0
                if i == 0 and j == 1 and k == 0:
                    count2 = 1
                if i == -1 and j == 0 and k == 0:
                    count2 = 0
                if i == 1 and j == 0 and k == 0:
                    count2 = 1
                if i == 0 and j == -1 and k == 0:
                    count2 = 0
                if i == 0 and j == 0 and k == 1:
                    count2 = 1
                # 直接做减法即可
                distance.append(abs(sphere_center[count1] - self.env.action_area[count2][count1]))
            # 如果超出了边界，直接返回距离为0，代表撞击上了
            if sphere_center[count1] <= self.env.action_area[0][1] or sphere_center[count1] >= self.env.action_area[1][count1]:
                return 0
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
        # state_grid = [self.x, self.y, self.z, dx, dy, dz,
        #               self.env.target.x, self.env.target.y, self.env.target.z,
        #               self.d_origin, self.step, self.distance,
        #               self.p_crash, self.now_bt, self.cost]
        state_grid = [self.x, self.y, self.z, dx, dy, dz,
                      self.env.target.x, self.env.target.y, self.env.target.z,
                      self.step / (4 * self.d_origin + 4 * self.env.action_area[1][2]),
                      self.distance]
        # state_grid = [dx, dy, dz]
        # state_grid = [self.x / self.env.action_area[1][0], self.y / self.env.action_area[1][1], self.z / self.env.action_area[1][2],
        #               dx / self.env.action_area[1][0], dy / self.env.action_area[1][1], dz / self.env.action_area[1][2],
        #               self.env.target.x / self.env.action_area[1][0], self.env.target.y / self.env.action_area[1][1], self.env.target.z / self.env.action_area[1][2],
        #               self.d_origin / math.sqrt(self.env.action_area[1][0] ** 2 + self.env.action_area[1][1] ** 2 + self.env.action_area[1][2] ** 2),
        #               self.step / (4 * self.d_origin + 4 * self.env.action_area[1][2]),
        #               self.distance / math.sqrt(self.env.action_area[1][0] ** 2 + self.env.action_area[1][1] ** 2 + self.env.action_area[1][2] ** 2),
        #               self.p_crash / 1.0, self.now_bt / 16, self.cost / self.bt]
        # state_grid = [self.x / self.env.action_area[1][0], self.y / self.env.action_area[1][1], self.z / self.env.action_area[1][2],
        #               dx / self.env.action_area[1][0], dy / self.env.action_area[1][1], dz / self.env.action_area[1][2],
        #               self.env.target.x / self.env.action_area[1][0], self.env.target.y / self.env.action_area[1][1], self.env.target.z / self.env.action_area[1][2],
        #               self.step / (4 * self.d_origin + 4 * self.env.action_area[1][2]),
        #               self.distance / self.d_origin]
        # 根据无人机周围的环境，更新无人机感受野中的情况
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # 尝试一下只看 上下前后左右6个传感器的
                    zero_num = 0
                    if i == 0:
                        zero_num += 1
                    if j == 0:
                        zero_num += 1
                    if k == 0:
                        zero_num += 1
                    if zero_num != 2 and k != 0:
                        # 将上下左右前后、中间层的四个角保留，其他跳过
                        continue
                    # 将无人机原点扣掉
                    if i == 0 and j == 0 and k == 0:
                        continue
                    # 初始化最短距离
                    nearest_distance_normal = 1000
                    nearest_distance = nearest_distance_normal
                    # 初始化碰撞的列表
                    collision_list = []
                    # 判断与建筑物的最近距离
                    for building in self.env.bds:
                        # 计算距离建筑物的最近距离
                        collision, build_distance = self.check_collision(np.array([self.x + i, self.y + j, self.z + k]),
                                                                         self.agent_r, building.left_down, building.right_up, i=i, j=j, k=k)
                        # 无人机最上方的传感器，只需要判断与活动区域天花板的距离即可
                        if i == 0 and j == 0 and k == 1:
                            build_distance = nearest_distance_normal
                        collision_list.append(collision)
                        if build_distance <= nearest_distance:
                            # 更新最近距离
                            nearest_distance = build_distance
                    if zero_num != 2:
                        state_grid.append(nearest_distance)
                        # 将同一层的四个角上的传感器直接跳过，它们不需要探测到边界的范围
                        continue
                    # 计算与边界的最近距离
                    limit_distance = self.col_limit_distance(np.array([self.x + i, self.y + j, self.z + k]), i=i, j=j, k=k)
                    # if (True in collision_list) or limit_distance == 0:
                    #     state_grid.append(1)
                    # else:
                    #     state_grid.append(0)
                    if limit_distance <= nearest_distance:
                        nearest_distance = limit_distance
                    # state_grid.append(nearest_distance_normal - nearest_distance)
                    state_grid.append(nearest_distance)
                    # state_grid.append((nearest_distance_normal - nearest_distance) / nearest_distance_normal)
                    # detector_distance = math.sqrt((self.x + i - self.env.target.x) ** 2 +
                    #                               (self.y + j - self.env.target.y) ** 2 +
                    #                               (self.z + k - self.env.target.z) ** 2)
                    # detector_distance = math.sqrt((self.x + i - self.env.target.x) ** 2 +
                    #                               (self.y + j - self.env.target.y) ** 2 +
                    #                               (self.z + k - self.env.target.z) ** 2) / self.d_origin
                    # state_grid.append(detector_distance)
        # 得到无人机的状态
        return state_grid

    def update(self, action):
        """
        无人机状态等更新函数
        :param action:无人机选择的动作
        :return:奖励，是否结束一轮迭代，额外信息
        """
        # 个人认为每次应该给它赋初值，不然就是上一个状态的最近距离
        self.nearest_distance = 20
        """返回参数"""
        done = False
        info = 4
        """相关参数"""
        b = 3  # 撞毁参数 原3
        wx = 0.5
        wy = 0.5
        wz = 0.5  # 爬升参数 原0.07
        we = 0.2  # 能量损耗参数  原0.2 ，0
        crash = 3  # 坠毁概率惩罚增益倍数 原3 ，0
        """计算无人机坐标变更值"""
        dx = action[0]
        dy = action[1]
        dz = action[2]
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
        if self.nearest_distance >= 3:
            # 撞毁概率等于0
            self.p_crash = 0
        else:
            # 根据公式计算撞毁概率
            # b:撞毁参数；增加一个1e-5是为了防止出现除以0的现象
            self.p_crash = 1 - math.exp(- b / (self.nearest_distance + 1e-2))
        # 计算爬升奖励    wc：爬升系数
        r_climb = -wx * abs(self.x - self.env.target.x) - wy * abs(self.y - self.env.target.y) - wz * abs(self.z - self.env.target.z)
        # 最小距离奖励
        if self.nearest_distance >= 3:
            r_n_distance = 0
        elif self.nearest_distance >= 1:
            # 让这个奖励绝对值大于1
            # r_n_distance = -10 / (self.nearest_distance + 1e-2)
            r_n_distance = - (1 - (self.nearest_distance / 3) ** 0.4) * 10
        else:
            r_n_distance = -10
        # 计算目标奖励，感觉有可能是目标奖励不明显，如果动作量接近的非常小的话
        # 感觉应该更改一下目标奖励的计算方式
        if self.distance > 1:
            r_target = 10 * (self.d_origin / self.distance) * Ddistance
        # 如果距离太近，已经是1了
        else:
            r_target = 10 * self.d_origin
        """计算总奖励r"""
        # 爬升奖励+目标奖励+能耗奖励-坠毁系数*坠毁概率
        # reward = (r_climb + r_target + r_e - crash * self.p_crash + r_n_distance) * 1e-1
        reward = 5 * (Ddistance - (abs(dx) + abs(dy) + abs(dz))) + 0.1 * self.nearest_distance
        # print("没有经过加减的reward:{}".format(reward))
        # reward = r_climb + r_target + r_e - crash * self.p_crash
        self.r_climb.append(r_climb)
        self.r_target.append(r_target)
        self.r_e.append(r_e)
        self.c_p_crash.append(- crash * self.p_crash)
        self.r_n_distance.append(r_n_distance)
        self.now_state.append([self.x, self.y, self.z])
        # print(self.p_crash)
        # print("reward:{}".format(reward))
        """终止状态判断"""
        # 如果无人机动作幅度过小，给予大量惩罚，但是可以继续运行
        if math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) <= 0.1:
            reward -= 10
            done = False
            info = 4
        if (self.x <= 1 or self.x >= self.env.action_area[1][0] - 1
                or self.y <= 1 or self.y >= self.env.action_area[1][1] - 1
                or self.z <= 1 or self.z >= self.env.action_area[1][2] - 1
                or True in collision_or_not):
            # 发生碰撞，产生巨大惩罚
            # 根据碰撞点距离目标的远近来设置奖励的大小
            if self.distance <= 10:
                reward -= 100
                done = True
                info = 2
            else:
                reward -= 200
                done = True
                info = 2
        if self.distance <= 2:
            # 到达目标点，给予大量奖励
            reward += 200
            done = True
            info = 1
        # 注意平衡探索步长和碰撞的关系
        if self.step >= 6 * self.d_origin + 6 * self.env.action_area[1][2]:
            # 步数超过最差步长（2*初始距离+2*空间高度），给予惩罚
            reward -= 50
            done = True
            info = 5
        if self.cost > self.bt:
            # 电量耗尽，给予大量惩罚
            reward -= 50
            done = True
            info = 3
        """更新无人机的坐标值"""
        self.x += dx
        self.y += dy
        self.z += dz
        return reward, done, info


if __name__ == '__main__':
    pass