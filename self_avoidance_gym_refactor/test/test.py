#!/home/ldt/anaconda3/envs/deeplearning/bin/python
# -*- coding: utf-8 -*-
import sys
import os
# 关键：若不是指定的 conda 解释器，则用它重新 exec 当前脚本
CONDA_PY = "/home/ldt/anaconda3/envs/deeplearning/bin/python"
if os.path.exists(CONDA_PY) and os.path.realpath(sys.executable) != os.path.realpath(CONDA_PY):
    os.execv(CONDA_PY, [CONDA_PY, os.path.abspath(__file__), *sys.argv[1:]])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import hydra
import torch
from environment_gym_refactor.environment.staticEnvironment import UavAvoidEnv
import gymnasium
import torch
import wandb
from navigation.SAC import MTransSAC
from environment_gym_refactor.uav.uav import UAVInfo
from tqdm import tqdm
import matplotlib.pyplot as plt
from main.tools import cfg_get, load_checkPoint, switch_model_eval
# 修改字体设置，尝试多个常见的中文字体
from pylab import mpl
# 优先尝试 SimHei，如果不行则尝试 WenQuanYi Micro Hei (Linux常见), Microsoft YaHei (Windows), 或 DejaVu Sans (英文保底)
mpl.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Microsoft YaHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="test", version_base=None)
def test(cfg) -> None:
    """
    测试脚本
    加载训练好的模型并进行测试
    :param cfg: 配置文件
    :return: None
    """
    # 获取当前程序运行的上一级文件夹
    upDir = os.path.dirname(os.path.dirname(__file__))
    """初始化算法与环境"""
    navigationAlgorithm: MTransSAC = MTransSAC(cfg)  # 初始化算法
    env: UavAvoidEnv = gymnasium.make('UavAvoid-v0', cfg=cfg)  # 初始化环境
    env.unwrapped.uavNums = int(cfg_get(cfg, "testUavNums", 1))  # 先用一个UAV进行测试
    env.unwrapped.level = int(cfg_get(cfg, "envLevel", 1))  # 设置环境难度等级
    """加载模型"""
    checkPointDir = cfg_get(cfg, "wandb.checkPointDir", "checkPoints")  # 检查点目录
    checkPointDir = os.path.join(upDir, checkPointDir)  # 检查点路径
    loadCheckPointPath = os.path.join(checkPointDir, getattr(cfg, "loadModel", "lastest.pt"))  # 要加载的检查点路径
    # 加载检查点
    _ = load_checkPoint(loadCheckPointPath, navigationAlgorithm)
    # 将模型切换为eval模式
    switch_model_eval(navigationAlgorithm)
    """获取测试参数"""
    testEpisodes = int(cfg_get(cfg, "testEpisodes", 1))  # 测试轮次
    # 全局统计
    totalSuccess, totalCollision, totalPowerEmpty, totalOver = 0, 0, 0, 0
    """进行测试"""
    with torch.no_grad():
        with tqdm(total=testEpisodes, desc="Testing") as progessBar:
            for episode in range(testEpisodes):
                """环境重置"""
                states = env.reset()  # 获取状态
                episodeReturn = 0  # 批次的累计奖励
                doneCount = 0  # 完成的无人机个数（包括成功、碰撞、超过步长、耗尽能量）
                """进行每一步的动作"""
                while doneCount < int(cfg_get(cfg, "testUavNums", 1)):
                    doneCount = 0  # 清空计数器
                    # 针对每一个无人机对象
                    for uav in env.unwrapped.uavs:
                        if uav.done:
                            doneCount += 1
                            continue
                        state = states[uav.uavID]  # 获取当前无人机的状态（是一个字典，需要再送入网络中进行处理）
                        action, _ = navigationAlgorithm.take_action(state)  # 算法选择动作
                        nextStates, reward, uavDone, _, information = env.step((action, uav.uavID))  # 与环境交互
                        episodeReturn += reward  # 累计奖励
                        states[uav.uavID] = nextStates  # 更新状态
                        print("=" * 20)
                        print(f"ID:{uav.uavID}, Action: {action}")
                        print(f"ID:{uav.uavID}, Reward: {reward}")
                        print(f"ID:{uav.uavID}, uavX: {uav.position.x}, uavY: {uav.position.y}, uavZ: {uav.position.z}")
                        env.render()  # 渲染环境
                        plt.pause(0.1)  # 暂停0.1秒，以便观察
                """统计无人机的终止状态"""
                # 计数器
                statusCounters= {
                    UAVInfo.SUCCESS: 0,
                    UAVInfo.COLLISION: 0,
                    UAVInfo.POWER_EMPTY: 0,
                    UAVInfo.STEP_OVER: 0,
                }
                for uav in env.unwrapped.uavs:
                    statusCounters[uav.information] += 1
                # 更新全局统计
                totalSuccess += statusCounters[UAVInfo.SUCCESS]
                totalCollision += statusCounters[UAVInfo.COLLISION]
                totalPowerEmpty += statusCounters[UAVInfo.POWER_EMPTY]
                totalOver += statusCounters[UAVInfo.STEP_OVER]
                # 更新进度条
                progessBar.update(1)
                plt.pause(60)  # 暂停60秒，以便观察
    """打印结果"""
    print("\n============================测试汇总=================================")
    print(f"成功：{totalSuccess}")
    print(f"碰撞：{totalCollision}")
    print(f"耗尽能量：{totalPowerEmpty}")
    print(f"超过步长：{totalOver}")
    print("====================================================================\n")
                    




if __name__ == "__main__":
    test()