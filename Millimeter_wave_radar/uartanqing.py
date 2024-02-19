"""
现在来改进一下，尝试使用卡尔曼滤波来预测后一个状态的值
这里是主要用在无人机自主截击阶段的
用于云控平台调用的再另外写
"""

import serial  # 导入串口通信模块
import serial.tools.list_ports
import numpy as np
import time  # 导入时间模块
import struct  # 导入结构化数据处理模块
import math  # 导入数学模块
import json  # 导入JSON数据处理模块
import collections
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from datetime import datetime


def ByteToHex(bins):
    """
    定义一个将字节转换为十六进制字符串的函数
    :param bins: 输入的byte字节类型的数据
    :return: 输出是16进制字符串
    """
    return ''.join(["%02X" % x for x in bins]).strip()


class ReplayBuffer:
    """
    状态回放池
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, timestamp_millisecond):
        """
        将数据加入buffer中
        :param state: 目标无人机的经纬高坐标
        :param timestamp_millisecond: 目标无人机的时间戳（毫秒级）
        :return: 更新容器
        """
        self.buffer.append((state, timestamp_millisecond))

    def size(self):
        """
        读取目前缓冲区中数据的数量
        :return: 缓冲区中数据数量
        """
        return len(self.buffer)


if __name__ == "__main__":
    # 经验回放池大小
    buffer_size = 10
    # 实例化经验回访池
    replayer = ReplayBuffer(buffer_size)
    # 构建卡尔曼滤波器
    kf = KalmanFilter(dim_x=6, dim_z=3)
    """
    定义状态转移矩阵，下面这样定义意味着认为在两个点之间，是匀速在运动
    """
    kf.F = np.array([[1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    """
    定义观测矩阵，下面这样定义意味着探测器可以观测到目标的三维坐标，但是观测不到速度
    """
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
    """
    定义过程噪声和观测噪声的协方差矩阵
    """
    kf.Q *= 0.01  # 过程噪声，这里面可以使用dt来更改数值，要是效果不好可以考虑更改这一块
    kf.R *= 0.01  # 观测噪声
    """
    初始协方差矩阵，下面这样定义代表只和自己有关，速度和状态无关
    """
    kf.P *= 1
    """雷达读取数据"""
    # 检查是否有可用的串口设备
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("无串口设备。")
    else:
        print("可用的串口设备如下：")
        for comport in ports_list:
            print(list(comport)[0], list(comport)[1])

    # 打开串口通信，波特率为115200
    ser = serial.Serial('COM5', 115200, timeout=1)
    # 检查串口是否打开
    a = ser.isOpen()
    print(a)
    # 文件路径（暂时指定为'test.txt'，可以根据实际情况修改）
    # 根据现有知识，这里应该是配置雷达的文件
    path = r'D:\学习\研究生\安擎\毫米波雷达文件\test.txt'
    f = open(path, 'r', encoding='UTF-8')  # 以只读方式打开文件
    i = 1
    while i < 45:
        # 循环44次，将配置的文件通过串口发出，发送给板卡
        txt = f.readline()  # 逐行读取文件内容
        print(txt)
        time.sleep(0.2)
        ser.write(txt.encode())  # 将读取的文本数据编码并写入串口
        i = i + 1
    f.close()  # 关闭文件
    ser.close()  # 关闭串口
    # 打开另一个串口
    # 指定串口设备路径、波特率、超时时间和字节间超时时间
    ser1 = serial.Serial('COM6', 921600, timeout=0.002, inter_byte_timeout=0.0001)
    ser1.stopbits = 1  # 停止位数量为1，每个字节后有一个停止位
    ser1.bytesize = 8  # 每个字节大小设置为8位
    ser1.close()
    ser1.open()
    ser1.write("hello".encode())  # 向串口写入"hello"字符串

    while(1):
        ser1.flush()  # 清空缓冲区
        temp =ser1.read_all()  # 读取所有可用的数据
        temp = ByteToHex(temp)  # 将字节数据转换为十六进制字符串
        array = list(temp)  # 接收数据存成list，将字符串转换为字符列表。没有检测到内容就是空的，检测到内容为非空
        range1_list = []  # range1：范围
        azimuth_list = []  # azimuth：方位角
        elevation_list = []  # elevation：高度（可能是海拔）
        doppler_list = []  # doppler：多普勒
        # 如果temp的前16个字符等于"0201040306050807"帧报头的魔法词
        if temp[0:16] == "0201040306050807":
            print("=" * 500)
            # 计算数据长度
            # 如果没有采集到目标，那么array的长度就是128
            print("array-shape:{}".format(np.shape(array)))
            # int(array[112], 16)意思是，将array[112]以十六进制理解，然后以十进制整数展示
            # 下面这段的意思就是将一段十六进制的数，转换为10进制的
            length = (int(array[112], 16) * 16 + int(array[113], 16) +
                      int(array[114], 16) * 16 ** 3 + int(array[115], 16) * 16 ** 2)
            # 计算length1，length和length1的结果是一样的，完全可以直接替换
            length1 = int(array[114] + array[115] + array[112] + array[113], 16)
            print("tvl_normal:{}".format(array[112:116]))
            print("length:{}, length1:{}".format(length, length1))

            # 如果temp的第104到108个字符等于“0600”
            if temp[104:108] == "0600":
                # numpoint = (length1 - 16) / 32  # 计算点的数量
                numpoint = (length1 - 8) / 16  # 计算点的数量
                print("point_num:{}".format(numpoint))
                # numpoint = 1  # 设置点的数量为1
                i = 1
                while i <= 1:
                    # 现在的index都是120
                    index = 32 * (i - 1) + 120
                    print("点云{}：{}".format(i, array[120 + 32 * (i - 1): 120 + 32 * i]))
                    # 获取range1的值 range：范围
                    range1 = (array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5]
                              + array[index + 2] + array[index + 3] + array[index] + array[index + 1])
                    # 将range1的十六进制字符串转换为浮点数，涉及到相应的结束标准
                    range1_f = struct.unpack('!f', bytes.fromhex(range1))[0]
                    range1_list.append(range1_f)  # 将range1的值添加到range1_list列表中
                    index += 8
                    # 获取azimuth的值 azimuth：方位角
                    azimuth = array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5] + array[index + 2] + array[index + 3] + array[index] + array[index + 1]
                    # 将azimuth的十六进制字符串转换为浮点数
                    azimuth_f = struct.unpack('!f', bytes.fromhex(azimuth))[0]
                    # print(azimuth_f)
                    azimuth_list.append(azimuth_f)
                    index += 8
                    # 获取elevation的值 elevation:高度仰角
                    elevation = array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5] + array[index + 2] + array[index + 3] + array[index] + array[index + 1]
                    # 将elevation的十六进制字符串转换为浮点数
                    elevation_f = struct.unpack('!f', bytes.fromhex(elevation))[0]
                    # print(elevation_f)
                    elevation_list.append(elevation_f)
                    index += 8
                    # 获取doppler的值  doppler：多普勒
                    doppler = array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5] + array[index + 2] + array[index + 3] + array[index] + array[index + 1]
                    # 将doppler的十六进制字符串转换为浮点数
                    doppler_f = struct.unpack('!f', bytes.fromhex(doppler))[0]
                    # print(doppler_f)
                    print("doppler_f:{}".format(doppler_f))
                    doppler_list.append(doppler_f)

                    # 计算目标的各个参数
                    x = range1_f * math.cos(elevation_f) * math.sin(azimuth_f)  # 计算x坐标
                    y = range1_f * math.cos(elevation_f) * math.cos(azimuth_f)  # 计算y坐标
                    z = range1_f * math.sin(elevation_f)  # 计算z坐标
                    # 格式化输出x,y,z坐标
                    efflist = "%f %f %f" % (x, y, z)
                    print(efflist)
                    # 得到时间戳（秒级）
                    timestamp = datetime.timestamp(datetime.now())
                    # 将时间戳转换为毫秒级
                    timestamp_millisecond = timestamp * 1000
                    replayer.add(np.array([x, y, z]), timestamp_millisecond)  # 将目标的各个参数添加到经验回放池中

                    i = i + 1

                time.sleep(0.5)

        # 如果经验回放池里面的数据超过了需要的最小值
        if replayer.size() >= 10:
            state_start, timestamp_start = replayer.buffer[0]  # 取出第一个状态
            # 确定为初始状态
            kf.x = np.concatenate((state_start, np.array([0, 0, 0])), axis=0)  # 将状态拼接成一个向量
            # 进行卡尔曼滤波
            filtered_positions = []
            # 得到传感器中的数据
            noisy_positions = []
            for i in range(10):
                if i > 0:
                    dt = (replayer.buffer[i][1] - replayer.buffer[i - 1][1]) / 1000  # 计算时间差
                    kf.F[0, 3] = dt
                    kf.F[1, 4] = dt
                    kf.F[2, 5] = dt
                kf.predict()  # 预测
                kf.update(replayer.buffer[i][0])  # 更新
                filtered_positions.append(kf.x[0:3])  # 将预测的结果添加到filtered_positions列表中
                noisy_positions.append(replayer.buffer[i][0])  # 将实测的结果添加到noisy_positions列表中
            # 提取滤波后的位置坐标
            filtered_positions = np.array(filtered_positions)
            noisy_positions = np.array(noisy_positions)

            # 绘制结果
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2], label='Filtered Positions', marker='.')
            print(noisy_positions)
            ax.plot(noisy_positions[:, 0], noisy_positions[:, 1], noisy_positions[:, 2], label='Noisy Positions', marker='x')
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Kalman Filter for 3D Trajectory Prediction')
            plt.show()

            """是否继续"""
            pd = input("是否继续(y/n)")
            if pd == 'n':
                break
            else:
                continue




  
          
          
           
              


