"""
训练主函数
"""
import sys
import os
# 关键：若不是指定的 conda 解释器，则用它重新 exec 当前脚本
CONDA_PY = "/home/ldt/anaconda3/envs/deeplearning/bin/python"
if os.path.exists(CONDA_PY) and os.path.realpath(sys.executable) != os.path.realpath(CONDA_PY):
    os.execv(CONDA_PY, [CONDA_PY, os.path.abspath(__file__), *sys.argv[1:]])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import hydra
from tqdm import tqdm
import signal
import time
import numpy as np
import gymnasium
import environment_gym_refactor
from tools import ReplayBuffer
from environment_gym_refactor.environment.staticEnvironment import UavAvoidEnv
from environment_gym_refactor.uav.uav import UAVInfo, UAV
# 可以显示中文
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg) -> None:
    """
    训练的主函数，Hydra注入config配置
    :param cfg: 配置文件
    :return: None
    """
    """wandb相关配置"""
    # pass
    # import wandb
    """初始化"""
    # 初始化算法
    navigationAlgorithm = None
    env: UavAvoidEnv = gymnasium.make('UavAvoid-v0', cfg=cfg)  # 初始化环境
    replayBuffer = ReplayBuffer(getattr(cfg, "bufferSize", 5000))  # 初始化经验回放池 
    """wandb与checkPoint的整合"""
    # 断点恢复
    startEpisodeIndex = 0  # 初始化为第0个批次
    totalEpisodes = int(getattr(cfg, "numEpisodes", 5000))  # 训练总轮次
    """训练迭代"""
    try:
        with tqdm(total=totalEpisodes, desc="Training", initial=startEpisodeIndex) as progressBar:
            episodeIndex = startEpisodeIndex  # 迭代轮次
            while episodeIndex < totalEpisodes:
                """环境重置与初始化记录变量"""
                states = env.reset()  # 获取状态
                episodeReturn = 0  # 批次的累计奖励
                successCount = 0  # 成功到达目标点的无人机个数
                collisionCount = 0  # 发证碰撞的无人机个数
                overCount = 0  # 超过最大步长的无人机个数
                powerExhaustCount =0  # 耗尽能量的无人机个数
                doneCount = 0  # 完成的无人机个数（包括成功、碰撞、超过步长、耗尽能量）
                """进行每一步的动作"""
                while doneCount < cfg.uav.uavNums:
                    doneCount = 0  # 清空计数器
                    # 针对于每一个无人机对象
                    for uav in env.uavs:
                        if uav.done:
                            doneCount += 1
                            continue
                        state = states[uav.uavID]  # 获取当前无人机的状态（是一个字典，需要再送入网络中进行处理）
                        action = navigationAlgorithm.take_action(state)  # 算法选择动作
                        nextStates, reward, uavDone, _, information = env.step((action, uav.uavID))  # 与环境交互
                        episodeReturn += reward  # 累计奖励
                        replayBuffer.add(state[uav.uavID], action, reward, nextStates, uavDone)  # 放入经验回放池
                        states[uav.uavID] = nextStates  # 更新状态
                """统计无人机的终止状态"""
                # 计数器
                statusCounters = {
                    UAVInfo.SUCCESS: 0,
                    UAVInfo.COLLISION: 0,
                    UAVInfo.POWER_EMPTY: 0,
                    UAVInfo.STEP_OVER: 0,
                }
                for uav in env.uavs:
                    statusCounters[uav.information] += 1
                # 统计
                successCount = statusCounters[UAVInfo.SUCCESS]
                collisionCount = statusCounters[UAVInfo.COLLISION]
                powerExhaustCount = statusCounters[UAVInfo.POWER_EMPTY]
                overCount = statusCounters[UAVInfo.STEP_OVER]
                """更新网络"""
                # 达到经验回放池最低大小，则更新网络
                if replayBuffer.size() > max(getattr(cfg, "minimalSize", 32), getattr(cfg, "batchSize", 32)):
                    batchSize = getattr(cfg, "batchSize", 32)
                    # 从经验回放池中采样
                    batchStates, batchActions, batchRewards, batchNextStates, batchDones = replayBuffer.sample(batchSize)
                    navigationAlgorithm.update({"states": batchStates,
                                                "actions": batchActions,
                                                "rewards": batchRewards,
                                                "nextStates": batchNextStates,
                                                "dones": batchDones})
                    # 学习率更新
                    # pass








    finally:
        pass


    



if __name__ == '__main__':
    main()
