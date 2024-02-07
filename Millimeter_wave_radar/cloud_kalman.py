"""
该文件用于云控平台的卡尔曼滤波程序
"""
import numpy as np
import json
from filterpy.kalman import KalmanFilter


def cloud_kalman(data):
    num = len(data['data'])
    states = np.zeros((num, 3))
    # 将目标点的经纬高读取出来
    for i in range(num):
        states[i, 0] = data['data'][i]['longitude']
        states[i, 1] = data['data'][i]['latitude']
        states[i, 2] = data['data'][i]['altitude']
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
    # 定义初始状态
    state_start = states[0]
    kf.x = np.concatenate((state_start, np.zeros(3)), axis=0)
    # 存储卡尔曼滤波后的状态
    filtered_positions = []
    # 存储传感器中的数据
    noisy_positions = []
    # 进行卡尔曼滤波
    for i in range(num):
        if i > 0:
            # 需要确认一下这里是秒级，时间戳计算出来是秒级，如果不是的话需要换算
            dt = data['data'][i]['timestamp'] - data['data'][i - 1]['timestamp']
            print(dt)
            kf.F[0, 3] = dt
            kf.F[1, 4] = dt
            kf.F[2, 5] = dt
        kf.update(states[i])
        filtered_positions.append(kf.x[0:3])
        noisy_positions.append(states[i])
    # 预测一步
    for i in range(1):
        kf.predict()
        filtered_positions.append(kf.x[0:3])
    filtered_positions = np.array(filtered_positions)
    state_json = {"longitude": filtered_positions[-1, 0],
                  "latitude": filtered_positions[-1, 1],
                  "altitude": filtered_positions[-1, 2]}
    # 将数据转为json
    state_json = json.dumps(state_json, indent=2)
    return state_json


if __name__ == '__main__':
    json_path = r"D:\PythonProject\Drone_self_avoidance\Millimeter_wave_radar\example.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
    state = cloud_kalman(data)
    print(type(state))
    print(state)
