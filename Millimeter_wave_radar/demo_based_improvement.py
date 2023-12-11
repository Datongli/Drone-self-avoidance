"""
该文件用于从SDK的demo出发，尝试得到多目标的位置和多普勒速度等信息
现在是可以得到多目标的位置和多普勒速度信息，目测还是比较合理，现在可以试试怎么得到IWR1843的原始数据
"""


import serial  # 导入串口通信模块
import serial.tools.list_ports
import numpy as np
import tkinter  # 导入图形用户界面模块
import os  # 导入操作系统相关的模块
import tkinter.messagebox  # 导入用于弹出消息框的模块
import tkinter.filedialog  # 导入用于选择文件对话框的模块
import time  # 导入时间模块
import struct  # 导入结构化数据处理模块
import math  # 导入数学模块
import json  # 导入JSON数据处理模块
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from queue import Queue
import threading



def ByteToHex(bins):
    """
    定义一个将字节转换为十六进制字符串的函数
    :param bins: 输入的byte字节类型的数据
    :return: 输出是16进制字符串
    """
    return ''.join(["%02X" % x for x in bins]).strip()


def HexToDecimal(hex_str_list):
    """
    定义一个函数，将以列表形式呈现的16进制数按照小端编码方式转换为10进制数
    :param hex_str_list: 列表类型的16进制数，里面是字符
    :return: 转换完成的10进制数
    """
    hex_str = ''.join(hex_str_list)
    reversed_hex_str = ''.join(reversed([hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]))
    decimal_value_little_endian = int(reversed_hex_str, 16)
    return decimal_value_little_endian


def HexToFloat(array):
    """
    定义一个函数，将以列表形式呈现的16进制数按照小端编码方式转换为float类型的数据
    :param array: 列表类型的16进制数，里面是字符
    :return: 转换完成的float类型数据
    """
    # 将十六进制字符列表连接成一个字符串
    hex_str = (array[6] + array[7] + array[4] + array[5] + array[2] + array[3] + array[0] + array[1])
    # 使用struct解析为浮点数，小端字节序
    # float_value = struct.unpack('f', struct.pack('I', decimal_value))[0]
    float_value = struct.unpack('!f', bytes.fromhex(hex_str))[0]
    return float_value


# 绘制三维图的函数
def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        try:
            # 从队列中获取坐标数据
            coordinates = coordinate_queue.get()

            # 清空原有图像
            ax.cla()

            # 绘制三维坐标点
            ax.scatter(coordinates['x'], coordinates['y'], coordinates['z'])

            # 设置图表标题和坐标轴标签
            ax.set_title('Real-time 3D Plot')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            # 显示图表
            plt.pause(0.1)

        except Exception as e:
            print(f"Error plotting 3D graph: {e}")



if __name__ == "__main__":

    # 检查是否有可用的串口设备
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("无串口设备。")
    else:
        print("可用的串口设备如下：")
        for comport in ports_list:
            print(list(comport)[0], list(comport)[1])

    # 打开串口通信，波特率为115200
    ser = serial.Serial('COM6', 115200, timeout=1)
    # 检查串口是否打开
    a = ser.isOpen()
    print(a)
    # ser.write("hello".encode())
    # path = tkinter.filedialog.askopenfilename()
    # print(path)
    # 文件路径（暂时指定为'test.txt'，可以根据实际情况修改）
    # 根据现有知识，这里应该是配置雷达的文件
    path = r'D:\学习\研究生\安擎\毫米波雷达文件\demo_velocity.txt'
    f = open(path, 'r', encoding='UTF-8')  # 以只读方式打开文件
    i = 1
    while i < 35:
        # 循环34次，将配置的文件通过串口发出，发送给板卡
        txt = f.readline()  # 逐行读取文件内容
        print(txt)
        time.sleep(0.2)
        ser.write(txt.encode())  # 将读取的文本数据编码并写入串口
        i = i + 1
    f.close()  # 关闭文件
    ser.close()  # 关闭串口
    # 打开另一个串口
    # 指定串口设备路径、波特率、超时时间和字节间超时时间
    ser1 = serial.Serial('COM5', 921600, timeout=0.002, inter_byte_timeout=0.0001)
    ser1.stopbits = 1  # 停止位数量为1，每个字节后有一个停止位
    ser1.bytesize = 8  # 每个字节大小设置为8位"FF"
    # c=ser1.isOpen()
    # print(c)
    ser1.close()
    ser1.open()
    ser1.write("hello".encode())  # 向串口写入"hello"字符串

    while (1):
        # data = ser1.readline(100)
        # #data1=bytes.fromhex(data)
        # #hexShow(data)
        # print(data)
        # print(type(data))
        # ser1.flushInput()
        # ser1.flush()
        ser1.flush()  # 清空缓冲区
        temp = ser1.read_all()  # 读取所有可用的数据

        # flush()
        # print(temp, len(temp))
        # time.sleep(2)
        # print(temp[0])
        temp = ByteToHex(temp)  # 将字节数据转换为十六进制字符串
        array = list(temp)  # 接收数据存成list，将字符串转换为字符列表。没有检测到内容就是空的，检测到内容为非空
        # print("array:{}".format(array))
        # e = array[96:100]  # 截取列表的一部分

        # int(ff, 16) #16进制转换为10进制
        range1_list = []  # range1：范围
        azimuth_list = []  # azimuth：方位角
        elevation_list = []  # elevation：高度（可能是海拔）
        doppler_list = []  # doppler：多普勒
        # print(temp, len(temp))
        # 如果temp的前16个字符等于"0201040306050807"帧报头的魔法词
        if temp[0:16] == "0201040306050807":
            print("=" * 500)
            # 计算数据长度
            # 现在是不论怎样，都直接是912长度，456字节，
            # 有的时候甚至没有将检测到的目标找全，到456字节就截断了
            print("array-shape:{}".format(np.shape(array)))
            # 检测到对象的数量，和后面的length是对应的，length是以字节计数的，每个对象16个字节
            print("检测到对象数量：{},{}".format(array[56: 64], HexToDecimal(array[56: 64])))
            # 提取出检测到的对象的数量
            object_num = HexToDecimal(array[56: 64])
            # 用于存储坐标数据的队列
            coordinate_queue = Queue()
            # 提取出每一个对象的x,y,z坐标
            for i in range(object_num):
                if i > 24:
                    continue
                x_hex = array[96 + i * 32: 104 + i * 32]
                y_hex = array[104 + i * 32: 112 + i * 32]
                z_hex = array[112 + i * 32: 120 + i * 32]
                v_hex = array[120 + i * 32: 128 + i * 32]
                x_float = HexToFloat(x_hex)
                y_float = HexToFloat(y_hex)
                z_float = HexToFloat(z_hex)
                v_float = HexToFloat(v_hex)
                print("第{}个;x:{};y:{};z:{};v:{}".format(i +1, x_float, y_float, z_float, v_float))
            # # 将坐标数据放入队列
            # coordinate_queue.put({'x': x_list, 'y': y_list, 'z': z_list})
            # # 启动绘制线程
            # plot_thread = threading.Thread(target=plot_3d)
            # plot_thread.start()
            # plt.show()
            # # 模拟实时获取坐标数据的时间间隔
            # time.sleep(0.5)
