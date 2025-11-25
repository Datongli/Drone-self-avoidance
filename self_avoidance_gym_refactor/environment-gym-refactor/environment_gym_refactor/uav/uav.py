"""
该文件用于构建无人机
"""
from ..environment.coordinate import Coordinate  # 包内相对导入
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random
from enum import Enum
from ..uav.sensor import Sensor, Lidar2D, Lidar3D


class UAVInfo(Enum):
    """无人机状态枚举"""
    SUCCESS = 1  # 成功到达目标点
    COLLISION = 2  # 发生碰撞
    POWER_EMPTY = 3  # 电量耗尽
    STEP_OVER = 5  # 步数超限


class UAV:
    """
    无人机类
    """
    # 传感器的构造器/工厂函数注册表
    SENSOR_REGISTRY: dict[str, Sensor] = {
        "2d_lidar": lambda cfg: Lidar2D(cfg=cfg),
        "3d_lidar": lambda cfg: Lidar3D(cfg=cfg),
    }

    def __init__(self, cfg, uavID: int) -> None:
        """
        无人机类初始化函数
        :param cfg: 配置文件
        :param uavID: 无人机ID
        return: None
        """
        self.cfg = cfg  # 配置文件
        self.uavID: int = uavID  # uavID
        self.position: Coordinate | None = None  # 位置
        self.radius: int | float = getattr(cfg.uav, "radius", 1.0)  # 半径
        self.power: int | float = getattr(cfg.uav, "power", 5000)  # 电量
        self.powerConsumption: int | float = getattr(cfg.uav, "powerConsumption", 5)  # 基础能耗，能耗/步
        self.powerCoefficient: int | float = getattr(cfg.uav, "powerCoefficient", 0.2)  # 能耗系数
        self.currentEnergyConsumption: int | float = 0  # 当前状态能耗
        self.energyAlreadyConsumption: int | float = 0  # 已消耗能量
        self.steps: int = 0  # 步数
        self.crashProbability: float = 0  # 坠毁概率
        self.done: bool = False  # 终止标志
        self.reward: int | float = 0  # 无人机获得的奖励
        self.information: int = 4  # 无人机的状态描述代码
        # 根据配置创建传感器实例
        sensorFactory = self.SENSOR_REGISTRY.get(getattr(cfg.uav, "sensorType", "2d_lidar"))
        self.sensor: Sensor | None = sensorFactory(cfg) if sensorFactory else None

    def get_information(self, env: gym.Env, yaw: int| float=np.pi/2.0) -> dict:
        """
        获取无人机的与传感器获得的状态
        先简单列写，后续再按照环境的实际使用，进行优化
        :param env: 环境
        :param yaw: 无人机的偏航角度
        :return: 无人机的状态
        """
        """获取无人机的状态"""
        uavState = []
        # 无人机自身的三维坐标
        uavState.append(self.position.x)
        uavState.append(self.position.y)
        uavState.append(self.position.z)
        # 无人机与目标点的三维坐标差值
        uavState.append(env.targets[self.uavID].x - self.position.x)
        uavState.append(env.targets[self.uavID].y - self.position.y)
        uavState.append(env.targets[self.uavID].z - self.position.z)
        # 目标点的三维坐标
        uavState.append(env.targets[self.uavID].x)
        uavState.append(env.targets[self.uavID].y)
        uavState.append(env.targets[self.uavID].z)
        uavState = np.array(uavState)
        """获取传感器的状态"""
        sensorState = self.sensor.get_sensor_data(self, env, yaw)
        """组合成一个状态向量字典并返回"""
        return {"uavState": uavState, 
                "sensorState": sensorState}
    
    def step(self, action: np.array) -> None:
        """
        执行无人机的一步动作
        :param action: 无人机的动作
        :return: None
        """
        """获取动作并更新无人机状态"""
        # 获取动作
        dx, dy, dz = action
        # 更新无人机状态
        self.position.x += dx
        self.position.y += dy
        self.position.z += dz
        """更新无人机信息"""
        self.steps += 1  # 步数加1
        # 更新当前状态能耗
        self.currentEnergyConsumption = self.powerConsumption + self.powerCoefficient * (abs(dx) + abs(dy) + abs(dz))
        # 更新已消耗能量
        self.energyAlreadyConsumption += self.currentEnergyConsumption


if __name__ == '__main__':
    pass