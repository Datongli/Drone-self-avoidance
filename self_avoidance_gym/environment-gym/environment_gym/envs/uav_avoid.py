from typing import Any, TypeVar, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random
from typing import Optional
import matplotlib.pyplot as plt


class Building:
    """
    建筑物的类
    """

    def __init__(self, x: float, y: float, length: float, width: float, height: float,
                 left_down: list, right_up: list):
        self.x = x  # 建筑物中心的x坐标
        self.y = y  # 建筑物中心的y坐标
        self.length = length  # 建筑物x方向长度一半
        self.width = width  # 建筑物y方向宽度一半
        self.height = height  # 建筑物高度
        self.left_down = left_down  # 建筑物左下角点的坐标
        self.right_up = right_up  # 建筑物右上角点的坐标


class Target:
    """
    目标点的类
    """

    def __init__(self, x: int, y: int, z: int):
        self.x = x  # 目标点的x坐标
        self.y = y  # 目标点的y坐标
        self.z = z  # 目标点的z坐标


class UAV:
    """
    无人机的类
    """

    def __init__(self, x: int, y: int, z: int, agent_r: int | float, env):
        """
        初始化无人机类
        :param x:无人机x初始坐标
        :param y:无人机y初始坐标
        :param z:无人机z初始坐标
        :param env:环境
        """
        # 初始化无人机位置
        self.x = x
        self.y = y
        self.z = z
        # 得到无人机的半径大小
        self.agent_r = agent_r
        # 初始化无人机运动情况
        self.bt = 5000  # 无人机电量
        self.p_bt = 5  # 无人机基础能耗，能耗/步
        self.k_bt = 0.2  # 无人机能耗系数，暂定为2
        self.now_bt = 4  # 无人机当前状态能耗
        self.cost = 0  # 无人机已经消耗能量
        self.step = 1  # 无人机已走步数
        self.p_crash = 0  # 无人机坠毁概率
        self.done = False  # 终止状态
        self.nearest_distance = 1000  # 最近的障碍物的距离
        self.env = env  # 环境
        # 无人机初始状态距离目标点的距离
        self.d_origin = math.sqrt(
            (self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 无人机距离目标点的距离
        self.distance = math.sqrt(
            (self.x - self.env.target.x) ** 2 + (self.y - self.env.target.y) ** 2 + (self.z - self.env.target.z) ** 2)
        # 无人机从环境中得到的奖励
        self.reward = []
        # 无人机总奖励
        self.total_reward = 0
        # 一些想要被观测到的值
        self.info = 4
        self.action = []
        self.now_state = []
        self.num = 1


class UavAvoidEnv(gym.Env):
    # 重写metadata, pygame适用于二维图像，三维的不适用，因此说不得要更改
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, agent_r: float, action_area: np.array, action_bound: int | float, uavs_num: int, render_mode: Optional[str] = None):
        """
        无人机避障环境初始化的函数
        :param agent_r:无人机抽象为球体的半径
        :param action_area:避障环境的整体空间
        :param action_bound:每一步动作的最大值（xyz三个方向分开）
        :param uavs_num:训练无人机的数量
        :param render_mode:渲染模式，默认为“人类”观测模式
        """
        # 规定观测状态的最小最大值
        space_low = np.array([action_area[0][0], action_area[0][1], action_area[0][2],
                              -action_bound, -action_bound, -action_bound,
                              action_area[0][0], action_area[0][1], action_area[0][2], 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        space_high = np.array([action_area[1][0], action_area[1][1], action_area[1][2],
                               action_bound, action_bound, action_bound,
                               action_area[1][0], action_area[1][1], action_area[1][2], 1,
                               math.sqrt(action_area[1][0] ** 2 + action_area[1][1] ** 2 + action_area[1][2] ** 2),
                               1, action_area[1][0], 1, action_area[1][1], action_area[1][2], action_area[1][2],
                               action_area[1][1], 1, action_area[1][0], 1], dtype=np.float32)
        # 定义动作空间
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=(3,), dtype=np.float32)
        # 定义观测（状态）空间
        self.observation_space = spaces.Box(low=space_low, high=space_high, shape=(21,), dtype=np.float32)
        self.agent_r = agent_r  # 无人机半径
        self.target = Target(0, 0, 0)  # 目标点位初始化
        self.action_area = action_area  # 可以运动的空间（对角线的形式）
        self.level = 1  # 课程学习的水平（训练难度等级）
        self.uavs = []  # 无人机对象集合
        self.bds = []  # 建筑物对象集合
        self.state = []  # 无人机的状态
        self.uavs_num = uavs_num  # 环境中无人机的个数
        self.render_mode = render_mode  # 观测模式，默认为人类观测模式
        if self.render_mode == 'human':
            self.fig = plt.figure()  # 画布
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')  # 三维画布

    @staticmethod
    def _overlop(building: Building, x: int | float, y: int | float, z: int | float,
                 interval: int | float = 1.0) -> bool:
        """
        静态方法
        判断无人机和目标点生成时是否与建筑物重合
        :param building:
        :param x: 点x坐标
        :param y: 点y坐标
        :param z: 点z坐标
        :param interval: 判断是否重合的间隔
        :return: True为重合，False为不重合
        """
        if (building.left_down[0] - interval <= x <= building.right_up[0] + interval
                and building.left_down[1] - interval <= y <= building.right_up[1] + interval
                and building.left_down[2] <= z <= building.right_up[2] + interval):
            return True
        else:
            return False

    @staticmethod
    def _check_collision(sphere_center: np.array, sphere_radius: int | float, building: Building,
                         sensor_pos: tuple = (0, 0, 0)) -> tuple[bool, int | float]:
        """
        静态方法
        判断无人机和建筑物是否相撞，以及计算距离
        :param sphere_center:计算位置（x,y,z）
        :param sphere_radius:无人机抽象成球体的半径
        :param building:建筑物实例
        :param sensor_pos:传感器点位
        :return:是否相撞、距离
        """
        left_down, right_up = np.array(building.left_down), np.array(building.right_up)  # 建筑物左下角点和右上角点
        closest_point = np.clip(sphere_center, left_down, right_up)  # 计算无人机和建筑物的最近点
        if np.array_equal(sphere_center, closest_point):
            distance = 0  # 如果传感器已经在建筑物内，直接给距离置0
        else:
            is_special_sensor = abs(sum(sensor_pos)) == 1  # 检查是否为特殊传感器（上下前后左右）
            sensor_pos = np.array(sensor_pos)
            if is_special_sensor:
                # 处理特殊传感器的逻辑
                zero_indices = np.where(sensor_pos == 0)[0]  # 找到i,j,k中哪两个为0
                not_zero_indices = np.where(sensor_pos != 0)[0]  # 找到i,j,k中哪个不为0
                ijk_not_zero = sensor_pos[not_zero_indices[0]]  # 找到i,j,k中不为0的值
                # 只管传感器“正前方”的障碍物
                if sphere_center[zero_indices[0]] == closest_point[zero_indices[0]] \
                        and sphere_center[zero_indices[1]] == closest_point[zero_indices[1]] \
                        and (
                        (ijk_not_zero < 0 and sphere_center[not_zero_indices[0]] > closest_point[not_zero_indices[0]])
                        or (ijk_not_zero > 0 and sphere_center[not_zero_indices[0]] < closest_point[
                    not_zero_indices[0]])):
                    distance = np.linalg.norm(sphere_center - closest_point)  # 计算距离
                else:
                    distance = 1000  # 不是传感器“正前方”的不管
            else:
                # 同一层四个角的传感器，或者无人机本身
                # 无人机本身
                distance = np.linalg.norm(sphere_center - closest_point)  # 计算距离
                if len(np.where(sensor_pos == 0)[0]) != 3:
                    # 四个角的传感器
                    if distance <= sphere_radius + 0.5:
                        distance = 0
                    else:
                        distance = 1000
        return distance <= sphere_radius, distance

    @staticmethod
    def _col_limit_distance(sphere_center: np.array, sphere_radius: int | float, action_area: np.array,
                            sensor_pos: tuple = (0, 0, 0)) -> tuple[bool, int | float]:
        """
        静态方法
        判断无人机和六个边界的距离
        :param sphere_center: 无人机中心点
        :param sphere_radius: 无人机抽象成球体的半径
        :param action_area: 避障环境的限制
        :param sensor_pos:传感器点位
        :return:是否相撞，最近距离
        """
        sensor_pos = np.array(sensor_pos)
        # 先判断是否在界限内
        out_or_not = ((sphere_center >= action_area[0, :] + sphere_radius) & (sphere_center <= action_area[1, :] - sphere_radius)).all()
        if not out_or_not:
            distance = 0  # 如果超出界限，直接给距离置0
        else:
            not_zero_indices = np.where(sensor_pos != 0)[0]  # 找到i,j,k中哪个不为0
            if len(not_zero_indices) == 0:
                # 如果是无人机本身
                diffs = np.abs(action_area - sphere_center)  # 计算无人机和边界的差值 利用了numpy的广播机制
                distance = np.min(diffs)  # 计算最小差值
            else:
                # 如果是上下左右传感器
                not_zero_value = sensor_pos[not_zero_indices[0]]  # 找到i,j,k中不为0的值
                line = 0 if not_zero_value == -1 else 1  # 判断是要和动作空间的上边界比较还是下边界
                # 计算与边界的距离
                distance = abs(sphere_center[not_zero_indices[0]] - action_area[line][not_zero_indices[0]])
        return distance <= sphere_radius, distance

    def _get_obs(self, uav: UAV) -> list:
        """
        用于得到无人机状态的函数私有类函数，不建议外部使用
        :param uav: 无人机类的实例
        :return: 无人机的状态
        """
        # 得到三个坐标的差值
        dx = self.target.x - uav.x
        dy = self.target.y - uav.y
        dz = self.target.z - uav.z
        """初始化要返回的状态表格，后续在其基础上增加"""
        state_grid = [uav.x, uav.y, uav.z, dx, dy, dz,
                      self.target.x, self.target.y, self.target.z,
                      uav.step / (4 * uav.d_origin + 4 * self.action_area[1][2]), uav.distance]
        """根据无人机周围环境情况，更新无人机周围分布的10个传感器的状态"""
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # 将上下左右前后、中间层的四个角传感器保留，组成10个传感器
                    if (zero_num := (i == 0) + (j == 0) + (k == 0) != 2 and k != 0) or (i == 0 and j == 0 and k == 0):
                        continue
                    nearest_distance_normal = 1000  # 初始化最短距离
                    nearest_distance = nearest_distance_normal  # 赋值给中间变量
                    # 判断与建筑物的最近距离
                    for building in self.bds:
                        # 计算建筑物的最近距离（正方向上）
                        _, build_distance = self._check_collision(np.array([uav.x + i, uav.y + j, uav.z + k]),
                                                                  self.agent_r, building, (i, j, k))
                        # 更新最近距离
                        nearest_distance = build_distance if build_distance <= nearest_distance else nearest_distance
                    if (i == 0) + (j == 0) + (k == 0) != 2:
                        # 如果是四个角的传感器，则直接赋值，同时跳过后续判断与边界的距离
                        state_grid.append(nearest_distance)
                        continue
                    # 判断上下左右前后6个传感器与"正前方"边界的距离
                    _, limit_distance = self._col_limit_distance(np.array([uav.x + i, uav.y + j, uav.z + k]), self.agent_r, self.action_area, (i, j, k))
                    # 更新最近距离
                    nearest_distance = limit_distance if limit_distance <= nearest_distance else nearest_distance
                    state_grid.append(nearest_distance)
        return state_grid

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.array, Any]:
        """
        重置环境
        将按照课程学习的水平，重置建筑物
        重新给uavs_num个无人机分配起始位置
        重新设立目标点
        :param seed:随机数种子
        :param options:额外的操作（字典的形式），例如更改整体避障环境的范围等
        :return:ndarray类型的数组，代表着uavs_num无人机的起始状态
        """
        """参数进行作用"""
        super().reset(seed=seed)
        if options is None:
            pass
        else:
            # 如果想要修改避障环境的范围等
            self.action_area = options.get('action_area') if 'action_area' in options else self.action_area
        """清空无人机对象和建筑对象"""
        self.uavs = []  # 清空无人机对象
        self.bds = []  # 清空建筑对象
        """构建建筑物"""
        # 随机生成建筑物，其数量根据课程学习的难度生成，同时判断是否有重叠
        building_num = random.randint(self.level, self.level * 2)
        while True:
            # 判断是否达到要生成的障碍物的数量
            if len(self.bds) >= building_num:
                break
            """构建建筑物的前期准备"""
            x = random.uniform(20, self.action_area[1][0] - 10)  # 建筑物中心的x坐标
            y = random.uniform(10, self.action_area[1][1] - 10)  # 建筑物中心的y坐标
            length = random.uniform(1, 5)  # 建筑物x方向长度一半
            width = random.uniform(1, 5)  # 建筑物y方向宽度一半
            height = random.uniform(self.action_area[1][2] - 20, self.action_area[1][2] - 3)  # 建筑物的高度
            left_down = [x - length, y - width, 0]  # 建筑物左下角点的坐标
            right_up = [x + length, y + width, height]  # 建筑物右上角点的坐标
            """判断预备生成的建筑物是否和已经生成的建筑物有重叠现象"""
            if len(self.bds) == 0:
                # 如果是第一个建筑物，直接跳过判断
                self.bds.append(Building(x, y, length, width, height, left_down, right_up))  # 实例化第一个建筑物类并加入环境的建筑物列表中
            else:
                overlop = False  # 是否重叠的标志位
                for building in self.bds:
                    if (abs(x - building.x) >= length + building.length
                            or abs(y - building.y) >= width + building.width):
                        # 如果没有重叠现象
                        continue  # 继续判断预备生成建筑物和建筑物列表中的其余建筑物对象是否有重叠现象
                    else:
                        overlop = True  # 有重叠，重叠标志位置位
                        break  # 只要有一个重叠就不需要再进行判断
                if not overlop:
                    # 实例化预备生成的建筑物类并加入环境的建筑物列表中
                    self.bds.append(Building(x, y, length, width, height, left_down, right_up))
        """随机生成目标点的位置"""
        while True:
            x = random.randint(60, 90)  # 目标点的x坐标
            y = random.randint(10, 90)  # 目标点的y坐标
            z = random.randint(10, self.action_area[1][2] - 5)  # 目标点的z坐标
            in_build = False  # 目标点是否在建筑物内的标志位
            for building in self.bds:
                if self._overlop(building, x, y, z, self.agent_r):
                    in_build = True  # 如果目标点在建筑物内，则标志位置位
                    break
            if not in_build:
                self.target = Target(x, y, z)  # 实例化目标点类并赋值给环境中的目标点
                break
        """随机生成无人机的初始位置"""
        for uav_num in range(self.uavs_num):
            too_many_num = 0  # 为了防止陷入死循环的计数器
            while True:
                x = random.randint(10, 15)  # 无人机的x坐标
                y = random.randint(10, 90)  # 无人机的y坐标
                z = random.randint(6, 8)  # 无人机的z坐标
                in_build = False  # 无人机是否在建筑物内的标志位
                # 无人机和目标点连线上是否有建筑物的标志位，在课程学习难度较高时应该启用
                complex_flag = False
                if self.level <= 0:
                    complex_flag = True
                # 确保没有无人机没有生成在障碍物的区域
                for building in self.bds:
                    if self._overlop(building, x, y, z, 2):
                        in_build = True  # 如果无人机在建筑物内，则标志位置位
                        break
                if not in_build:
                    # 判断无人机和目标点连线上是否有障碍物
                    uav_local = np.array([x, y, z])  # 无人机坐标
                    tar_local = np.array([self.target.x, self.target.y, self.target.z])  # 目标点坐标
                    vector = tar_local - uav_local  # 计算两个点之间的距离向量
                    distance = np.linalg.norm(vector)  # 计算两个点之间的距离（给距离向量求模）
                    normed_vector = vector / distance  # 将向量标准化为单位向量
                    mini_distance = distance / 100.0  # 取100个点，离散化目标点和无人机之间的距离
                    for i in range(100):
                        if i == 99:
                            too_many_num += 1
                        if too_many_num >= 1000:
                            # 防止陷入死循环
                            complex_flag = True
                        vector_pd = normed_vector * (i * mini_distance) + uav_local
                        for building in self.bds:
                            if self._overlop(building, vector_pd[0], vector_pd[1], vector_pd[2], 0):
                                # 如果无人机和目标点直接有建筑物的遮挡，则将complex_flag置位
                                complex_flag = True
                    if complex_flag:
                        # 全部条件都满足，uav列表增加
                        uav = UAV(x, y, z, self.agent_r, self)
                        uav.num = uav_num + 1
                        self.uavs.append(uav)
                        break
        """返回uavs_num个无人机初始状态"""
        for uav in self.uavs:
            self.state.append(self._get_obs(uav))
        if self.render_mode == 'human':
            # 如果是“人类”观测模式，调用渲染函数
            self.render(1)  # 由于是第一次渲染，故需要渲染建筑物
        return np.array(self.state, dtype=np.float32), {}

    def step(self, u) -> tuple[list, int | float | Any, bool, bool, dict[str, int]]:
        """
        时间步函数，用于更新无人机的state
        :param u:无人机采取的动作和编号组成的元组
        :return:下一状态，奖励，是否结束，用于调试的额外信息等
        """
        """选定第i个无人机"""
        action, i = u  # 分别取出动作和无人机编号
        uav = self.uavs[i]  # 对于第i个无人机执行一个时间步的动作
        uav.nearest_distance = 1000  # 初始化最近的障碍物的距离
        """计算无人机坐标变更值"""
        dx, dy, dz = action  # 分别代表x,y,z三个方向的动作
        # 距离变化量
        Ddistance = uav.distance - math.sqrt(
            (uav.x - self.target.x) ** 2 + (uav.y - self.target.y) ** 2 + (uav.z - self.target.z) ** 2)
        collision_or_not = []  # 无人机是否和建筑物相撞的判断列表
        # 更新距离值
        uav.distance = math.sqrt(
            (uav.x - self.target.x) ** 2 + (uav.y - self.target.y) ** 2 + (uav.z - self.target.z) ** 2)
        uav.step += 1  # 更新无人机已走步数
        """计算能耗"""
        uav.cost += uav.now_bt  # 更新无人机已消耗能量
        uav.now_bt = uav.p_bt + uav.k_bt * (abs(dx) + abs(dy) + abs(dz))  # 更新无人机当前状态能耗
        """计算与建筑物之间的最短距离"""
        for building in self.bds:
            collision, build_distance = self._check_collision(np.array([uav.x, uav.y, uav.z]), self.agent_r, building)
            collision_or_not.append(collision)  # 将无人机是否和建筑物相撞的判断加入列表
            # 更新最近的障碍物的距离
            uav.nearest_distance = build_distance if build_distance <= uav.nearest_distance else uav.nearest_distance
        """计算与六个边界的距离"""
        limit_collision, limit_distance = self._col_limit_distance(np.array([uav.x, uav.y, uav.z]), self.agent_r, self.action_area)
        # 更新最近的与边界的距离
        uav.nearest_distance = limit_distance if limit_distance <= uav.nearest_distance else uav.nearest_distance
        """计算总奖励r"""
        reward = 5 * (Ddistance - abs(dx) - abs(dy) - abs(dz)) + 0.1 * uav.nearest_distance
        """返回参数"""
        # 初始化奖励，是否完成，用于训练的额外参数
        middle_tuples = (reward, False, 4)
        """记录一些观测值"""
        uav.now_state.append([uav.x, uav.y, uav.z])
        """终止状态判断"""
        if math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) <= 0.1:
            # 如果无人机动作幅度过小，认为其在原地打转，给予惩罚，但是可以继续运行
            middle_tuples = (reward - 10, False, 4)
        elif limit_collision or True in collision_or_not:
            # 发生碰撞，给予惩罚，同时停止运行
            multiple = 1 if uav.distance <= 10 else 2  # 惩罚倍数
            middle_tuples = (reward - 100 * multiple, True, 2)
        elif uav.distance <= 2:
            # 到达目标点，给予奖励，同时停止运行
            middle_tuples = (reward + 200, True, 1)
        elif uav.step >= 6 * uav.d_origin + 6 * self.action_area[1][2]:
            # 如果超过最大步长，给予惩罚，同时停止运行
            middle_tuples = (reward - 50, True, 5)
        elif uav.cost >= uav.bt:
            # 电量耗尽，给予惩罚，同时停止运行
            middle_tuples = (reward - 50, True, 3)
        """从中间元组中取出奖励，是否结束，额外信息"""
        reward, done, info = middle_tuples
        """更新无人机参数"""
        uav.x += dx  # 更新无人机的x坐标
        uav.y += dy  # 更新无人机的y坐标
        uav.z += dz  # 更新无人机的z坐标
        uav.info = info  # 更新无人机的额外信息
        self.uavs[i] = uav  # 更新第i个无人机
        """得到动作作用之后的状态"""
        next_state = self._get_obs(uav)
        """可视化"""
        if self.render_mode == 'human':
            self.render()
        Info = {"info": info}
        return np.array(next_state, dtype=np.float32), reward, done, False, Info

    def render(self, flag: int = 0) -> None:
        """
        渲染函数
        :return:渲染的图像|图像（数值）列表|None
        """
        if flag == 1:
            self.ax.clear()  # 清除画布上的内容，重新开始
            plt.ion()
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
                # 绘制目标坐标点
                self.ax.scatter(self.target.x, self.target.y, self.target.z, c='red')
        for uav in self.uavs:
            self.ax.scatter(uav.x, uav.y, uav.z, c='blue')
        plt.show(block=False)


if __name__ == '__main__':
    pass
