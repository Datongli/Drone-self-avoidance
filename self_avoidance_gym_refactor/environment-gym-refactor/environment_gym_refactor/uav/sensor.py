"""
该文件用于构建无人机的传感器
"""
from abc import ABC, abstractmethod
# from ..uav.uav import UAV
import gymnasium as gym
import numpy as np
# from ..environment.staticEnvironment import UavAvoidEnv


class Sensor(ABC):
    """传感器基类"""
    def __init__(self, cfg) -> None:
        """
        传感器类初始化函数
        :param cfg: 配置文件
        return: None
        """
        self.cfg = cfg  # 配置文件

    @abstractmethod
    def get_sensor_data(self, *args, **kwargs) -> any:
        """
        获取传感器数据
        :return: any
        """
        pass


class Lidar2D(Sensor):
    """2D激光雷达传感器"""
    def __init__(self, cfg) -> None:
        """
        2D激光雷达传感器初始化函数
        :param cfg: 配置文件
        return: None
        """
        super(Lidar2D, self).__init__(cfg)
        """从配置中获取参数"""
        self.numBeams: int = getattr(cfg.uav.sensor, "numBeams", 512)  # 激光束数量
        self.minRange: int | float = getattr(cfg.uav.sensor, "minRange", 0.5)  # 最近探测距离
        self.maxRange: int | float = getattr(cfg.uav.sensor, "maxRange", 100.0)  # 最远探测距离
        self.sensorData: np.ndarray | None = None  # 传感器数据

    def get_sensor_data(self, uav, env: gym.Env, yaw: float=np.pi/2.0) -> np.ndarray:
        """
        仿真2D激光雷达传感器在当前环境下的探测数据
        雷达在无人机所在水平面上扫描，扫描范围为[-pi, pi]
        从无人机“正后方”开始扫描，共numBeams个激光束
        :param uav: 无人机实例
        :param env: 环境实例
        :param yaw: 无人机当前偏航角度，单位为弧度，以x轴为0度，逆时针为正旋转
        :return: 2D激光雷达传感器在当前环境下的探测数据
        """
        """获取无人机当前的位置"""
        uavX: float = uav.position.x  # 无人机x坐标
        uavY: float = uav.position.y  # 无人机y坐标
        uavZ: float = uav.position.z  # 无人机z坐标
        """构建每条雷达激光束"""
        # 初始化每条雷达激光束为最大探测距离，后面会被更新为实际探测距离
        ranges: np.ndarray = np.full(self.numBeams, self.maxRange, dtype=np.float32)
        # 为每条激光束计算方向
        thetas: np.array = np.linspace(-np.pi, np.pi, self.numBeams, endpoint=False, dtype=np.float32)
        directions: np.ndarray = np.zeros((self.numBeams, 2), dtype=np.float32)  # 激光束方向向量
        # 计算每个激光束的全局方向向量
        for beamIndex, theta in enumerate(thetas):
            # 计算当前激光束的全局角度，从无人机当前偏航角的正后方开始，逆时针旋转360度
            angle: float = yaw + theta
            # 将角度转换为单位方向向量
            directionX: float = np.cos(angle)
            directionY: float = np.sin(angle)
            directions[beamIndex, 0] = directionX
            directions[beamIndex, 1] = directionY
        """处理环境中静态障碍物"""
        for building in getattr(env, "staticObstacles", []):
            # 跳过和无人机不在同一水平面上的障碍物
            if not (uavZ <= building.height):
                continue
            # 从建筑物中取得x方向上的最小/最大坐标
            boxMinX: float = building.leftDown.x
            boxMaxX: float = building.rightUp.x
            # 从建筑物中取得y方向上的最小/最大坐标
            boxMinY: float = building.leftDown.y
            boxMaxY: float = building.rightUp.y
            # 对于每一条激光束，计算与该建筑物投影矩形的交点距离
            for beamIndex in range(self.numBeams):
                # 激光束方向向量
                directionX: float = directions[beamIndex, 0]
                directionY: float = directions[beamIndex, 1]
                # 计算激光束与该建筑物投影矩形的最近交点距离
                hitDistance: float | None = self._ray_intersection_distance(
                    pointX=uavX,
                    pointY=uavY,
                    dircetionX=directionX,
                    directionY=directionY,
                    boxMinX=boxMinX,
                    boxMaxX=boxMaxX,
                    boxMinY=boxMinY,
                    boxMaxY=boxMaxY,
                )
                # 若没有交点，则继续检查下一条射线或下一个障碍物
                if hitDistance is None:
                    continue
                # 如果有交点，则更新激光束的探测距离为最近的交点距离
                ranges[beamIndex] = min(ranges[beamIndex], hitDistance)
        """处理环境边界"""
        # 获取世界边界
        worldMinX, worldMinY = 0.0, 0.0
        worldMaxX, worldMaxY = getattr(self.cfg.env, "length", 100.0), getattr(self.cfg.env, "width", 100.0)
        for beamIndex in range(self.numBeams):
            # 激光束方向向量
            directionX: float = directions[beamIndex, 0]
            directionY: float = directions[beamIndex, 1]
            # 计算射线与整个世界边界矩形的交点距离
            hitDistance: float | None = self._ray_intersection_distance(
                pointX=uavX,
                pointY=uavY,
                dircetionX=directionX,
                directionY=directionY,
                boxMinX=worldMinX,
                boxMaxX=worldMaxX,
                boxMinY=worldMinY,
                boxMaxY=worldMaxY,
            )
            # 若没有交点，则继续检查下一条射线或下一个障碍物
            if hitDistance is None:
                continue
            # 如果有交点，则更新激光束的探测距离为最近的交点距离
            ranges[beamIndex] = min(ranges[beamIndex], hitDistance)
        """处理并返回结果"""
        ranges = np.clip(ranges, self.minRange, self.maxRange)
        self.sensorData = ranges
        return ranges
            
    @staticmethod
    def _ray_intersection_distance(pointX: float, pointY: float, dircetionX: float, directionY: float,
                                   boxMinX: float, boxMaxX: float, boxMinY: float, boxMaxY: float) -> float | None:
        """
        使用AABB算法
        计算激光束与矩形的最近交点距离
        P0代表激光束起点坐标，t代表射线的距离，D代表激光束方向向量
        :param pointX: 激光束起点x坐标
        :param pointY: 激光束起点y坐标
        :param dircetionX: 激光束x方向分量
        :param directionY: 激光束y方向分量
        :param boxMinX: 矩形x方向最小坐标
        :param boxMaxX: 矩形x方向最大坐标
        :param boxMinY: 矩形y方向最小坐标
        :param boxMaxY: 矩形y方向最大坐标
        :return: 激光束与矩形的最近交点距离，若激光束与矩形无交点则返回None
        """
        """初始化在x,y方向上的参数区间"""
        # t 表示射线参数：P(t) = P0 + t * D
        tMin: float = -np.inf  # 初始化在x方向上的参数区间为负无穷大
        tMax: float = np.inf  # 初始化在x方向上的参数区间为正无穷大
        """处理x方向上与矩形的相交"""
        # 若激光束x方向分量近似为0，则射线与矩形无交点
        if abs(dircetionX) < 1e-8:
            # 起点在矩形的x范围之外，则与矩形无交点
            if pointX < boxMinX or pointX > boxMaxX:
                return None
        # 根据射线方程（pointX + t * directionX）计算与矩形左右边界交点的t值
        else:
            t1: float = (boxMinX - pointX) / dircetionX  # 计算与矩形左边界交点的t值
            t2: float = (boxMaxX - pointX) / dircetionX  # 计算与矩形右边界交点的t值
            # 更新tMin和tMax，取较大的较小值和较小的较大值
            tMin: float = max(tMin, min(t1, t2))
            tMax: float = min(tMax, max(t1, t2))
        """处理y方向上与矩形的相交"""
        # 若激光束y方向分量近似为0，则射线与矩形无交点
        if abs(directionY) < 1e-8:
            # 起点在矩形的y范围之外，则与矩形无交点
            if pointY < boxMinY or pointY > boxMaxY:
                return None
        # 根据射线方程（pointY + t * directionY）计算与矩形上下边界交点的t值
        else:
            t3: float = (boxMinY - pointY) / directionY  # 计算与矩形下边界交点的t值
            t4: float = (boxMaxY - pointY) / directionY  # 计算与矩形上边界交点的t值
            # 更新tMin和tMax，取较大的较小值和较小的较大值
            tMin: float = max(tMin, min(t3, t4))
            tMax: float = min(tMax, max(t3, t4))
        """判断结果并返回结果"""
        # 如果最大t小于0，说明整个矩形在射线的反方向上，无交点
        if tMax < 0:
            return None
        # 选取实际的交点参数tHit
        # 如果tMin >= 0，则说明最近的进入点是tMin
        # 如果tMin < 0，则射线七点在矩形内部，最近离开点是tMax
        if tMin >= 0:
            tHit: float = tMin
        else:
            tHit: float = tMax
        # 如果最终的tHit小于0，则认为无有效交点
        if tHit < 0:
            return None
        return float(tHit)


class Lidar3D(Sensor):
    """3D激光雷达传感器"""
    def __init__(self, cfg) -> None:
        """
        3D激光雷达传感器初始化函数
        :param cfg: 配置文件
        return: None
        """
        super(Lidar3D, self).__init__(cfg)

    def get_sensor_data(self, *args, **kwargs):
        pass


class DepthCamera(Sensor):
    """深度相机传感器"""
    def __init__(self, cfg) -> None:
        """
        深度相机传感器初始化函数
        :param cfg: 配置文件
        return: None
        """
        super(DepthCamera, self).__init__(cfg)

    def get_sensor_data(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    pass