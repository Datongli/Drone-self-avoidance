"""
这个文件是丁建新师兄发来的，处理毫米波雷达信息的文件，现在先试图看懂
可以尝试更改一下
"""

import serial  # 导入串口通信模块
import serial.tools.list_ports
import tkinter  # 导入图形用户界面模块
import os  # 导入操作系统相关的模块
import tkinter.messagebox  # 导入用于弹出消息框的模块
import tkinter.filedialog  # 导入用于选择文件对话框的模块
import time  # 导入时间模块
import struct  # 导入结构化数据处理模块
import math  # 导入数学模块
import json  # 导入JSON数据处理模块


def ByteToHex(bins):
    """
    定义一个将字节转换为十六进制字符串的函数
    :param bins: 输入的byte字节类型的数据
    :return: 输出是16进制字符串
    """
    return ''.join(["%02X" % x for x in bins]).strip()


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
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    # 检查串口是否打开
    a = ser.isOpen()
    print(a)
    # ser.write("hello".encode())
    # path = tkinter.filedialog.askopenfilename()
    # print(path)
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
    ser1 = serial.Serial('/dev/ttyACM1', 921600, timeout=0.002, inter_byte_timeout=0.0001)
    ser1.stopbits = 1  # 停止位数量为1，每个字节后有一个停止位
    ser1.bytesize = 8  # 每个字节大小设置为8位
    # c=ser1.isOpen()
    # print(c)
    ser1.close()
    ser1.open()
    ser1.write("hello".encode())  # 向串口写入"hello"字符串

    while(1):
        # data = ser1.readline(100)
        # #data1=bytes.fromhex(data)
        # #hexShow(data)
        # print(data)
        # print(type(data))
        # ser1.flushInput()
        # ser1.flush()
        ser1.flush()  # 清空缓冲区
        temp =ser1.read_all()  # 读取所有可用的数据

        # flush()
        # print(temp)
        # time.sleep(2)
        # print(temp[0])
        temp = ByteToHex(temp)  # 将字节数据转换为十六进制字符串
        # index=str.find("0201040306050807")
        # print(temp)
        array = list(temp)  # 接收数据存成list，将字符串转换为字符列表
        e = array[96:100]  # 截取列表的一部分

        # int(ff, 16) #16进制转换为10进制
        range1_list = []  # range1：范围
        azimuth_list = []  # azimuth：方位角
        elevation_list = []  # elevation：高度（可能是海拔）
        doppler_list = []  # doppler：多普勒
        # 如果temp的前16个字符等于"0201040306050807"
        if temp[0:16] == "0201040306050807":
            # 计算数据长度
            # int(array[112], 16)意思是，将array[166]以十六进制理解，然后以十进制整数展示
            length = (int(array[112], 16) * 16 + int(array[113], 16) +
                      int(array[114], 16) * 16 ** 3 + int(array[115], 16) * 16 ** 2)
            # 计算长度1
            length1 = int(array[114] + array[115] + array[112] + array[113], 16)

            # 如果temp的第104到108个字符等于“0600”
            if temp[104:108] == "0600":
                numpoint = (length1 - 16) / 32  # 计算点的数量
                numpoint = 1  # 设置点的数量为1
                i = 1
                while i <= numpoint:
                    index = 32 * (i - 1) + 120
                    # 获取range1的值 range：范围
                    range1 = (array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5]
                              + array[index + 2] + array[index + 3] + array[index] + array[index + 1])
                    # 将range1的十六进制字符串转换为浮点数
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
                    # 获取elevation的值 elecation:海拔高度
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
                    doppler_list.append(doppler_f)
                    i = i + 1
                x = range1_f * math.sin(azimuth_f)  # 计算x坐标
                y = range1_f * math.cos(azimuth_f)  # 计算y坐标
                z = range1_f * math.sin(elevation_f)  # 计算z坐标
                # 格式化输出x,y,z坐标
                efflist="%f %f %f" %(x,y,z)
                print(efflist)
                # 构造post_data字典
                post_data = {"Obstacle":
                    {
                    "Obstacle_X": float(x),
                    "Obstacle_Y": float(y),
                    "Obstacle_Z": float(z),
                 }
                }
                print(json.dumps(post_data))
                #os.system(
                # f"python3 drone_sdk.py -f linkkit_thing_post_property -j '{json.dumps(post_data)}' -s 0")
                time.sleep(0.5)

  
          
          
           
              


