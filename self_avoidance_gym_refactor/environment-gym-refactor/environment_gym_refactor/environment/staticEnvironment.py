"""
静态环境
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random
import os
from typing import Optional
import matplotlib
# 在vscode debugpy调试环境下，改用非交互后端，避免Qt相关错误
# if os.environ.get("PYDEVD_USE_FRAME_EVAL", None) is not None:
#     matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..uav.uav import UAV  # 包内相对导入
from ..environment.target import Target
from ..obstacle.building import Building
from ..environment.coordinate import Coordinate


class UavAvoidEnv(gym.Env):
    """
    无人机自主避障环境类
    """
    # 重写metadata属性
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg) -> None:
        """
        无人机自主避障环境类初始化函数
        :param cfg: 配置文件
        return: None
        """
        """为了通过gymnasium对自定义环境的规范检查，需要重写以下的属性"""
        # 定义动作空间
        self.action_space = spaces.Box(low=-getattr(cfg, "actionBound", 1), high=getattr(cfg, "actionBound", 1), shape=(3,), dtype=np.float32)
        # 定义观测空间
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(getattr(cfg.uav.sensor, "numBeams", 512)+9,), dtype=np.float32)
        self.cfg = cfg  # 配置文件
        """初始化环境参数"""
        self.length: int | float = getattr(cfg.env, "length", 100)  # 环境长
        self.width: int | float = getattr(cfg.env, "width", 100)  # 环境宽
        self.height: int | float = getattr(cfg.env, "height", 25)  # 环境高
        self.level: int = getattr(cfg.env, "level", 1)  # 环境难度等级
        """初始化目标点参数"""
        self.targets: list[Target] = []  # 目标点列表
        self.targetRadius: int | float = getattr(cfg.env, "targetRadius", 0.5)  # 目标点半径
        """初始化静态障碍物参数"""
        self.staticObstacles: list[Building] = []  # 静态障碍物列表
        self.obstacleHorizontalRange: int| float = getattr(cfg.env, "obstacleHorizontalRange", 10)  # 障碍物最大水平半径限制
        self.obstacleVerticalRange: int| float = getattr(cfg.env, "obstacleVerticalRange", 25)  # 障碍物垂直高度限制
        """初始化无人机参数"""
        self.uavNums: int = getattr(cfg.uav, "uavNums", 1)  # 无人机数量
        self.uavs: list[UAV] = []  # 无人机列表
        """观测模式"""
        self.renderMode: str = getattr(cfg, "renderMode", "human")  # 渲染模式
        if self.renderMode == "human":
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

    def reset(self) -> list:
        """
        重置环境
        :return: list
        """
        """清空目标点、静态障碍物、无人机"""
        self.targets.clear()
        self.staticObstacles.clear()
        self.uavs.clear()
        """生成障碍物、目标点、无人机"""
        self._static_obstacles_generate()
        self._targets_generate()
        self._uavs_generate()
        """返回状态观测值"""
        if self.renderMode == "human":
            # 如果是“human”模式，渲染环境
            self.render(1)
        return self._state_observation()
        
    def step(self, u: tuple) -> tuple | None:
        """
        针对指定的无人机与动作，返回新的状态观测值、奖励、是否结束、其他信息
        :param u: 元组，包含动作和无人机ID
        :return: tuple | None
        """
        """选定无人机"""
        action, uavID = u  # 获取动作和无人机ID
        uav: UAV = self.uavs[uavID]  # 选定第i个无人机
        # 没执行动作之前无人机距离目标点之间的距离
        prevDistanceToTarget = math.sqrt(
                                     (uav.position.x - self.targets[uavID].x) ** 2 +
                                     (uav.position.y - self.targets[uavID].y) ** 2 +
                                     (uav.position.z - self.targets[uavID].z) ** 2)
        """更新无人机状态"""
        uav.step(action)
        # 计算执行动作之后与目标点之间的距离
        currentDistanceToTarget = math.sqrt((uav.position.x - self.targets[uavID].x) ** 2 +
                                     (uav.position.y - self.targets[uavID].y) ** 2 +
                                     (uav.position.z - self.targets[uavID].z) ** 2)
        # 传感器数据获取
        sensorData = None
        if hasattr(uav, "sensor") and uav.sensor is not None:
            try:
                sensorData = uav.sensor.get_sensor_data(uav, self, yaw=np.pi/2.0)
            except Exception as e:
                print(f"获取传感器数据时出错:{e}")
        """计算奖励值"""
        # 1 日常奖励部分
        # 靠近目标奖励
        targetReward = getattr(self.cfg.env.reward, "wTarget", 0.1) * (prevDistanceToTarget - currentDistanceToTarget)
        # 角度引导奖励
        vecToTarget = np.array([
            self.targets[uavID].x - uav.position.x,
            self.targets[uavID].y - uav.position.y,
            self.targets[uavID].z - uav.position.z
        ])  # 计算理想位移向量
        vecAction = np.array(action)  # 计算实际的动作
        normTarget = np.linalg.norm(vecToTarget) + 1e-6  # 计算向量的模
        normAction = np.linalg.norm(vecAction) + 1e-6  # 获取动作的模
        cosineSimilarity = np.dot(vecAction, vecToTarget) / (normTarget * normAction)  # 计算余弦相似度
        wAngle = getattr(self.cfg.env.reward, "wAngle", 0.5)  # 角度引导权重
        angleReward = wAngle * cosineSimilarity  # 计算角度奖励
        # 边界逼近惩罚
        boundaryPenalty = 0.0  # 初始化边界逼近惩罚
        boundSafeDistance = getattr(self.cfg.env.reward, "safeDistance", 4.0)  # 边界安全距离
        wBound = getattr(self.cfg.env.reward, "wBound", 0.1)  # 边界惩罚权重
        minDistanceToWall = min(abs(uav.position.x), abs(uav.position.y), abs(uav.position.z), self.length - abs(uav.position.x),
                                 self.width - abs(uav.position.y), self.height - abs(uav.position.z))  # 到边界的最小距离
        if minDistanceToWall < boundSafeDistance:  # 如果到边界的距离小于安全距离
            boundaryPenalty = -wBound * (boundSafeDistance - minDistanceToWall)
        # 障碍物惩罚
        obstaclePenalty = 0.0  # 初始化障碍物惩罚
        if sensorData is not None and len(sensorData) > 0:
            sensorData = (np.array(sensorData) * (getattr(self.cfg.uav.sensor, "maxRange", 100.0) - getattr(self.cfg.uav.sensor, "minRange", 0.5))
                          + getattr(self.cfg.uav.sensor, "minRange", 0.5))  # 反归一化传感器数据
            safeDistance = getattr(self.cfg.env.reward, "safeDistance", 5.0)  # 安全距离
            dangerThreshold = getattr(self.cfg.env.reward, "dangerThreshold", 3.0)  # 危险阈值
            wMin = getattr(self.cfg.env.reward, "wMin", 0.5)  # 最小距离项权重
            wDanger = getattr(self.cfg.env.reward, "wDanger", 0.3)  # 危险束比例项权重
            wField = getattr(self.cfg.env.reward, "wField", 0.2)  # 距离势场项权重
            alpha = getattr(self.cfg.env.reward, "alpha", 2.0)  # 势场塑形的指数
            minDistance = float(np.min(sensorData))  # 最小距离
            dangerRatio = float(np.mean(sensorData < dangerThreshold))  # 危险束比例
            fieldTerm = float(np.mean((np.maximum(0.0, safeDistance - sensorData) / safeDistance) ** alpha))  # 距离势场项
            # 归一化minDistance惩罚
            minDistanceTerm = (safeDistance - min(minDistance, safeDistance)) / safeDistance
            minDistanceTerm = max(0.0, minDistanceTerm)
            # 计算障碍物惩罚
            obstaclePenalty = wMin * minDistanceTerm + wDanger * dangerRatio + wField * fieldTerm
            obstaclePenalty = -obstaclePenalty  # 转化为负数
            # 如果最小距离小于危险阈值，则取消角度引导奖励
            if minDistance < dangerThreshold:
                angleReward = 0.0
        # 能量消耗惩罚
        wEnergy = getattr(self.cfg.env.reward, "wEnergy", 0.01)
        energyPenalty = -wEnergy * uav.currentEnergyConsumption
        # 高度保护奖励
        z = uav.position.z
        heightMin, heightMax = 0.0, self.height  # 高度范围
        safeMargin = getattr(self.cfg.env.reward, "safeMargin", 3.0)  # 保护距离
        altitudePenalty = 0.0  # 初始化高度保护奖励
        wAltitude = getattr(self.cfg.env.reward, "wAltitude", 0.5)  # 高度保护奖励系数
        if z < safeMargin:
            altitudePenalty = - wAltitude * ((safeMargin - z) / safeMargin) ** 2
        elif z > (heightMax - safeMargin):
            altitudePenalty = - wAltitude * ((z - (heightMax - safeMargin)) / safeMargin) ** 2
        # 汇总日常奖励
        shapingReward = targetReward + obstaclePenalty + energyPenalty + boundaryPenalty + angleReward + altitudePenalty
        # 2 终止奖励部分
        terminalReward = 0.0  # 初始化终止奖励
        targetRadius = getattr(self.cfg.env, "targetRadius", 1.0)  # 目标点半径
        maxSteps = getattr(self.cfg.env, "maxSteps", int(2 * (self.length + self.width + self.height)))  # 最大步数
        powerEmpty = uav.energyAlreadyConsumption >= uav.power  # 无人机能量是否为空
        # 碰撞检测
        collision = self._collision_detection(uav)  
        # 终止分类
        if collision:
            uav.done = True
            uav.information = 2
            terminalReward = -getattr(self.cfg.env.reward, "collisionPenalty", 10.0)
        elif currentDistanceToTarget <= targetRadius:
            uav.done = True
            uav.information = 1
            terminalReward = getattr(self.cfg.env.reward, "targetReward", 20.0)
        elif powerEmpty:
            uav.done = True
            uav.information = 3
            terminalReward = -getattr(self.cfg.env.reward, "energyOutPenalty", 8.0)
        elif uav.steps >= maxSteps:
            uav.done = True
            uav.information = 5
            terminalReward = -getattr(self.cfg.env.reward, "stepOutPenalty", 5.0)
        # 总奖励
        reward = shapingReward + terminalReward
        # 获取状态观测值
        nextState = uav.get_information(self, np.pi/2.0)
        """可视化"""
        if self.renderMode == "human":
            self.render()
        return nextState, reward, uav.done, False, {"uavInformation": uav.information}

    def render(self, flag: int=0) -> None:
        """
        渲染函数
        :param flag: 渲染标志位
        :return: None
        """
        if flag == 1:
            self.ax.clear()
            plt.ion()
            # 绘制出所有的障碍物
            for obstacle in self.staticObstacles:
                x = obstacle.x
                y = obstacle.y
                z = 0
                dx = obstacle.halfX
                dy = obstacle.halfY
                dz = obstacle.height
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
            # 绘制出目标点
            for target in self.targets:
                self.ax.scatter(target.x, target.y, target.z, c='red')
        # 绘制出所有的无人机
        for uav in self.uavs:
            self.ax.scatter(uav.position.x, uav.position.y, uav.position.z, c='blue')
        plt.show(block=False)

    def _static_obstacles_generate(self) -> None:
        """
        生成静态障碍物
        :return: None
        """
        """确定障碍物的数量"""
        # 按照课程学习的思路，生成与当前等级对应的静态障碍物数量
        staticObstaclesNumber = random.randint(self.level, self.level * 2)
        """循环生成静态障碍物"""
        count = 0  # 防止陷入无限循环
        while len(self.staticObstacles) < staticObstaclesNumber and count < 5000:
            # 构建建筑物前期准备
            x = random.uniform(self.length * 0.1, self.length * 0.9)  # 建筑物中心的x坐标
            y = random.uniform(self.width * 0.1, self.width * 0.9)  # 建筑物中心的y坐标
            halfX = random.uniform(self.obstacleHorizontalRange * 0.1, self.obstacleHorizontalRange)  # 建筑物x方向长度一半
            halfY = random.uniform(self.obstacleHorizontalRange * 0.1, self.obstacleHorizontalRange)  # 建筑物y方向宽度一半
            height = random.uniform(self.obstacleVerticalRange * 0.1, self.obstacleVerticalRange)  # 建筑物的高度
            leftDown = Coordinate(x - halfX, y - halfY, 0)  # 建筑物左下角的坐标
            rightUp = Coordinate(x + halfX, y + halfY, height)  # 建筑物右上角的坐标
            # 创建障碍物
            generateObstacle = Building(x, y, halfX, halfY, height, leftDown, rightUp)
            # 检查是否与已有的障碍物重叠
            if not self.staticObstacles or not self._is_building_overlap(generateObstacle):
                self.staticObstacles.append(generateObstacle)
            count += 1

    def _is_building_overlap(self, generateObstacle: Building) -> bool:
        """
        检查静态障碍物是否与已存在的静态障碍物重叠
        :param generateObstacle: 静态障碍物
        :return: bool
        """
        for building in self.staticObstacles:
            # 根据建筑物的中心坐标和长度宽度判断是否重叠
            if (abs(generateObstacle.x - building.x) < generateObstacle.halfX + building.halfX and
                abs(generateObstacle.y - building.y) < generateObstacle.halfY + building.halfY):
                return True
        return False
    
    def _targets_generate(self) -> None:
        """
        生成目标点
        :return: None
        """
        """确定目标点的数量"""
        targetsNumber = self.uavNums
        """循环生成目标点"""
        count = 0  # 防止陷入无限循环
        while len(self.targets) < targetsNumber:
            # 构建目标点前期准备
            x = random.uniform(self.length * 0.2, self.length * 0.8)  # 目标点的x坐标
            y = random.uniform(self.width * 0.5, self.width * 0.8)  # 目标点的y坐标
            z = random.uniform(self.height * 0.4, self.height * 0.6)  # 目标点的z坐标
            # 创建目标点
            generatorTargent = Target(x, y, z)
            # 检查是否与障碍物距离过近
            if count < 50000:
                if not self._is_target_overlop(generatorTargent):
                    self.targets.append(generatorTargent)
            else:
                # 紧急情况，空间太拥挤，强制生成
                self.targets.append(generatorTargent)
            count += 1

    def _is_target_overlop(self, generatorTargent: Target) -> bool:
        """
        检查目标点是否与障碍物距离过近
        :param generatorTargent: 目标点
        :return: bool
        """
        for building in self.staticObstacles:
            # 根据建筑物的中心坐标和半径判断是否重叠
            if (abs(generatorTargent.x - building.x) < (self.targetRadius + building.halfX) * 2 and
                abs(generatorTargent.y - building.y) < (self.targetRadius + building.halfY) * 2):
                return True
        return False
    
    def _uavs_generate(self) -> None:
        """
        生成无人机
        :return: None
        """
        """确定无人机的数量"""
        uavNums = self.uavNums
        """循环生成无人机"""
        count = 0  # 防止陷入无限循环
        while len(self.uavs) < uavNums and count < 5000:
            # 构建无人机前期准备
            x = random.uniform(self.length * 0.2, self.length * 0.8)  # 无人机的x坐标
            y = random.uniform(self.width * 0.05, self.width * 0.1)  # 无人机的y坐标
            z = random.uniform(self.height * 0.2, self.height * 0.25)  # 无人机的z坐标
            # 创建无人机
            generatorUav = UAV(self.cfg, len(self.uavs))
            generatorUav.position = Coordinate(x, y, z)
            # 检查是否与障碍物距离过近
            if count < 50000:
                if not self._is_uav_overlop(generatorUav):
                    self.uavs.append(generatorUav)
            else:
                # 紧急情况，空间太拥挤，强制生成
                self.uavs.append(generatorUav)
            count += 1

    def _is_uav_overlop(self, generatorUav: UAV) -> bool:
        """
        检查无人机是否与障碍物距离过近
        :param generatorUav: 无人机
        :return: bool
        """
        for building in self.staticObstacles:
            # 根据建筑物的中心坐标和半径判断是否重叠
            if (abs(generatorUav.position.x - building.x) < (generatorUav.radius + building.halfX) * 1.5 and
                abs(generatorUav.position.y - building.y) < (generatorUav.radius + building.halfY) * 1.5):
                return True
        return False
    
    def _state_observation(self) -> list:
        """
        返回状态观测值
        :return: list
        """
        """初始化要返回的状态观测值列表"""
        states: list = []
        """每一个无人机都返回一个状态观测值"""
        for uav in self.uavs:
            states.append(uav.get_information(self, np.pi/2.0))
        return states
    
    def _collision_detection(self, uav: UAV) -> bool:
        """
        碰撞检测
        :param uav: 无人机
        :return: bool
        """
        """检查无人机是否越界"""
        if not ((0 + uav.radius < uav.position.x < self.length - uav.radius) and
                (0 + uav.radius < uav.position.y < self.width - uav.radius) and
                (0 + uav.radius < uav.position.z < self.height - uav.radius)):
            return True
        """检查无人机是否与障碍物发生碰撞"""
        for obstacle in self.staticObstacles:
            if ((obstacle.leftDown.x - uav.radius < uav.position.x < obstacle.rightUp.x + uav.radius) and
                (obstacle.leftDown.y - uav.radius < uav.position.y < obstacle.rightUp.y + uav.radius) and
                (obstacle.leftDown.z - uav.radius < uav.position.z < obstacle.rightUp.z + uav.radius)):
                return True
        return False


if __name__ == '__main__':
    pass