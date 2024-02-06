import math
import time
import struct
import math
import threading
import argparse
import os
import serial
import numpy as np
import collections
from filterpy.kalman import KalmanFilter


class ReplayBuffer:
    """
    状态回放池，用于存储雷达传感器读取到，转换为经纬高坐标系下的位置数据
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state):
        self.buffer.append((state))

    def size(self):
        return len(self.buffer)


# 毫米波雷达连接类
class SerialConnect(object):
    def __init__(self) -> None:
        # 初始化属性
        self.__serial_connect = None
        self.connected = False
        self.need_coordinate = False
        self.need_kalman = False
        self.obstacle_x = 0
        self.obstacle_y = 0
        self.obstacle_z = 0
        self.obstacle_distance = -1
        self.obstacle_time = 0

        # 连接毫米波雷达
        self.__connect_thread = threading.Thread(target=self.serial_connect)
        self.__connect_thread.start()
        # 实例化经验回放池，暂定为10个点
        self.replayer = ReplayBuffer(10)


    # 初始化毫米波雷达
    @classmethod
    def serial_init(cls):
        print("drone_sdk_serial init start")
        serial_0_connected = False
        connection_failed_time = 0
        while not serial_0_connected:
            try:
                # serial_connect_0 = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
                serial_connect_0 = serial.Serial('COM5', 115200, timeout=1)
            except:
                pass
            else:
                file = open(r'D:\学习\研究生\安擎\毫米波雷达文件\test.txt', "r", encoding="UTF-8")
                for _ in range(1, 45):
                    str = file.readline()
                    time.sleep(0.1)
                    serial_connect_0.write(str.encode())
                file.close()
                serial_connect_0.close()
                serial_0_connected = True
                print("drone_sdk_serial init complete")

    # 连接毫米波雷达
    def serial_connect(self):
        os.system("sudo systemctl start drone_sdk_serial")

        connection_failed_time = 0
        while not self.connected:
            try:
                self.__serial_connect = serial.Serial(
                    'COM6', 921600, timeout=0.002, inter_byte_timeout=0.0001
                )
            except:
                pass
            else:
                self.__serial_connect.stopbits = 1
                self.__serial_connect.bytesize = 8
                # self.__serial_connect.close()
                # self.__serial_connect.open()
                # self.__serial_connect.write("hello".encode())
                self.connected = True

    # 坐标转换
    def coordinate_transformation(self, o_longitude, o_latitude, o_altitude, yaw, x, y, z):
        """
        用于目标坐标转换的函数
        :param o_longitude:智能无人机的经度，精度为1e-7，单位：度
        :param o_latitude: 智能无人机的纬度，精度为1e-7，单位：度
        :param o_altitude: 智能无人机的海拔，单位：米
        :param yaw: 智能无人机的偏航角，参考正北方向，单位：度
        :param x: 目标相对于智能无人机的x坐标，单位：米
        :param y: 目标相对于智能无人机的y坐标，单位：米
        :param z: 目标相对于智能无人机的z坐标，单位：米
        :return: 经过转换的目标经纬高坐标
        """
        # 计算目标与无人机的水平距离
        r = math.sqrt(x ** 2 + y ** 2)
        # 计算目标与y轴的夹角，同时转化为角度制
        alpha = math.degrees(math.atan(x / y))
        # 计算目标与正北方向的夹角，同时转换为弧度制
        theta = math.radians(alpha + yaw)
        # 计算目标的经纬高坐标，注意除以基准值
        # 注意一下东西经度会不会有问题，再确认一下，还有就是米转换为经度和纬度时，可能会有问题
        target_longitude = o_longitude + r * math.sin(theta) / (85.39 * 1e3)
        target_latitude = o_latitude + r * math.cos(theta) / (111 * 1e3)
        target_altitude = o_altitude + z
        return target_longitude, target_latitude, target_altitude

    # 轮询
    def loop(self, time_interval, o_longitude, o_latitude, o_altitude, yaw) -> bool:
        if not self.connected:
            return False

        if time_interval < 0.3:
            return False

        self.__serial_connect.flush()
        temp = self.__serial_connect.read_all()
        temp = "".join(["%02X" % x for x in temp]).strip()
        array = list(temp)

        if temp[0:16] == "0201040306050807":
            if temp[104:108] == "0600":
                index = 120
                range1 = (
                    array[index + 6]
                    + array[index + 7]
                    + array[index + 4]
                    + array[index + 5]
                    + array[index + 2]
                    + array[index + 3]
                    + array[index]
                    + array[index + 1]
                )
                range1_f = struct.unpack("!f", bytes.fromhex(range1))[0]
                index = index + 8
                azimuth = (
                    array[index + 6]
                    + array[index + 7]
                    + array[index + 4]
                    + array[index + 5]
                    + array[index + 2]
                    + array[index + 3]
                    + array[index]
                    + array[index + 1]
                )
                azimuth_f = struct.unpack("!f", bytes.fromhex(azimuth))[0]
                index = index + 8
                elevation = (
                    array[index + 6]
                    + array[index + 7]
                    + array[index + 4]
                    + array[index + 5]
                    + array[index + 2]
                    + array[index + 3]
                    + array[index]
                    + array[index + 1]
                )
                elevation_f = struct.unpack("!f", bytes.fromhex(elevation))[0]
                """更改了计算坐标的值的方法"""
                self.obstacle_x = range1_f * math.cos(elevation_f) * math.sin(azimuth_f)
                self.obstacle_y = range1_f * math.cos(elevation_f) * math.cos(azimuth_f)
                self.obstacle_z = range1_f * math.sin(elevation_f)
                self.obstacle_distance = math.sqrt(
                    pow(self.obstacle_x, 2)
                    + pow(self.obstacle_y, 2)
                    + pow(self.obstacle_z, 2)
                )
                self.obstacle_time = time.time()
                # 坐标转换，存入状态回放池
                target_longitude, target_latitude, target_altitude = self.coordinate_transformation(o_longitude, o_latitude, o_altitude, yaw,
                                                                                                    self.obstacle_x, self.obstacle_y, self.obstacle_z)
                self.replayer.add(np.array([target_longitude, target_latitude, target_altitude]))
                if self.obstacle_distance < 100:
                    self.need_coordinate = True
                # 如果状态回放池里的长度满足算法所需要的长度
                if self.replayer.size() >= 10:
                    self.need_kalman = True
                    # print(f"need_coordinate: {self.obstacle_distance}")
                # else:
                #     self.need_coordinate = False
                # print(
                #     f"time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.obstacle_time))}, x: {format(self.obstacle_x, ' .2f')}, y: {format(self.obstacle_y, ' .2f')}, z: {format(self.obstacle_z, ' .2f')}, distance: {format(self.obstacle_distance, ' .2f')}"
                # )

        return True

    # 规划路线函数 路径起始点, 路径终点, 无人机与障碍物安全距离, 每一步步长
    def coordinate(self, start_point, end_point, distance=0.00008):
        delta_0 = math.atan2(
            (end_point[1] - start_point[1]), (end_point[0] - start_point[0])
        )
        new_point_1 = [
            start_point[0] + math.cos(delta_0 + math.pi / 3) * distance,
            start_point[1] + math.sin(delta_0 + math.pi / 3) * distance,
            start_point[2],
        ]
        new_point_2 = [
            new_point_1[0] + math.cos(delta_0) * distance,
            new_point_1[1] + math.sin(delta_0) * distance,
            start_point[2],
        ]
        new_point_3 = [
            new_point_2[0] + math.cos(delta_0 - math.pi / 3) * distance,
            new_point_2[1] + math.sin(delta_0 - math.pi / 3) * distance,
            start_point[2],
        ]

        track_points = []
        track_points.append(
            {
                "Latitude": new_point_1[0],
                "Longitude": new_point_1[1],
                "Altitude": new_point_1[2],
            }
        )
        track_points.append(
            {
                "Latitude": new_point_2[0],
                "Longitude": new_point_2[1],
                "Altitude": new_point_2[2],
            }
        )
        track_points.append(
            {
                "Latitude": new_point_3[0],
                "Longitude": new_point_3[1],
                "Altitude": new_point_3[2],
            }
        )
        return track_points

    # 输出目标无人机下一个位置状态的函数
    # 为了精确度的考虑，不适合预测太多步之后的
    def kalman_filtering(self):
        # 实例化卡尔曼滤波器
        kf = KalmanFilter(dim_x=6, dim_z=3)
        # 定义状态转移矩阵
        kf.F = np.array([[1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        # 定义观测矩阵
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        # 定义过程噪声和观测噪声的协方差矩阵
        kf.Q *= 0.01  # 过程噪声，这里面可以使用dt来更改数值，要是效果不好可以考虑更改这一块
        kf.R *= 0.01  # 观测噪声
        # 取出初始状态
        state_start = self.replayer.buffer[0]
        kf.x = np.concatenate((state_start, np.zeros(3)), axis=0)
        # 存储卡尔曼滤波后的状态
        filtered_positions = []
        # 存储传感器中的数据
        noisy_positions = []
        # 进行卡尔曼滤波
        for i in range(len(self.replayer.buffer)):
            kf.predict()
            kf.update(self.replayer.buffer[i])  # 更新
            filtered_positions.append(kf.x[0:3])
            noisy_positions.append(self.replayer.buffer[i])
        # 预测一步
        for i in range(1):
            kf.predict()
            filtered_positions.append(kf.x[0:3])
        filtered_positions = np.array(filtered_positions)
        # 返回最后预测的一步的经纬高坐标
        return filtered_positions[-1]


# 执行脚本
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="无人机雷达SDK")
    parser.add_argument("--fun", "-f", help="方法名", required=False)
    args = parser.parse_args()

    if args.fun == "init":
        SerialConnect.serial_init()
    else:
        SerialConnect.serial_init()
        # 初始化守护进程
        daemon = SerialConnect()
        # 开始轮询
        while True:
            daemon.loop(0.3, 40.3586406, 116.7446954, 100, 30)
            print(f"connected: {daemon.connected}")
            if daemon.need_coordinate:
                print(f"need_coordinate: {daemon.obstacle_distance}")
                daemon.need_coordinate = False
            if daemon.need_kalman:
                coordinate = daemon.kalman_filtering()
                print("coordinate: {}".format(coordinate))
                daemon.need_kalman = False
            time.sleep(0.3)
