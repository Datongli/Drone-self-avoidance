"""
训练主函数
"""
import sys
import os
# 关键：若不是指定的 conda 解释器，则用它重新 exec 当前脚本
# CONDA_PY = "/home/ldt/anaconda3/envs/deeplearning/bin/python"
# if os.path.exists(CONDA_PY) and os.path.realpath(sys.executable) != os.path.realpath(CONDA_PY):
#     os.execv(CONDA_PY, [CONDA_PY, os.path.abspath(__file__), *sys.argv[1:]])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../environment-gym-refactor')))
import hydra
from hydra.utils import to_absolute_path
from tqdm import tqdm
import gymnasium
import torch
import wandb
from tools import ReplayBuffer, draw_rewards_images, cfg_get, load_checkPoint, wandb_init
from tools import save_checkPoint, upload_wandb
from environment_gym_refactor.environment.staticEnvironment import UavAvoidEnv
from environment_gym_refactor.uav.uav import UAVInfo
from navigation.SAC import MTransSAC
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
    # 获取当前程序运行的上一级文件夹
    upDir = os.path.dirname(os.path.dirname(__file__))
    """wandb初始化"""
    wbEnabled = bool(cfg_get(cfg, "wandb.enabled", True))  # 是否使用wandb
    wandb_init(cfg, upDir)
    saveEvery = int(cfg_get(cfg, "wandb.saveEvery", 10))  # 每训练n轮保存一次模型
    """初始化算法与环境"""
    navigationAlgorithm: MTransSAC = MTransSAC(cfg)  # 初始化算法
    env: UavAvoidEnv = gymnasium.make('UavAvoid-v0', cfg=cfg)  # 初始化环境
    replayBuffer = ReplayBuffer(getattr(cfg, "bufferSize", 10000))  # 初始化经验回放池 
    rewardList = []  # 记录的每轮次累计奖励
    """wandb与checkPoint的整合"""
    # 准备checkPoint的存储目录
    checkPointDir = cfg_get(cfg, "wandb.checkPointDir", "checkPoints")  # 检查点目录
    checkPointPathConfig = cfg_get(cfg, "wandb.checkPointPath", "checkPoints/kw9xf9k7.pt")  # 检查点路径配置
    checkPointDir = os.path.join(upDir, checkPointDir)  # 检查点路径
    os.makedirs(checkPointDir, exist_ok=True)  # 创建检查点目录
    lastestCheckPointPath = os.path.join(checkPointDir, "lastest.pt")  # 最新检查点路径
    # 断点恢复
    wbResumeFlag = cfg_get(cfg, "wandb.resumeFlag", True)  # 是否从上次断点继续
    if wbResumeFlag:
        # 优先使用latest.pt
        loadFromPath = lastestCheckPointPath if os.path.exists(lastestCheckPointPath) else checkPointPathConfig
        if loadFromPath and os.path.exists(loadFromPath):
            try:
                # 加载检查点并返回episode索引
                episodeIndex = load_checkPoint(loadFromPath, navigationAlgorithm)
            except Exception as e:
                episodeIndex = 0  # 若加载失败，重置为0
                print(f"[ERROR] 恢复检查点失败，将从头开始训练：{e}")
        else: pass
    else: episodeIndex = 0  # 若未启用恢复，重置为0
    env.unwrapped.level = navigationAlgorithm.difficultyLevel  # 设置环境难度
    # 用于参数变化曲线的记录
    watchRegistered = False  # 是否已注册watch
    totalEpisodes = int(getattr(cfg, "numEpisodes", 5000))  # 训练总轮次
    """训练迭代"""
    try:
        with tqdm(total=totalEpisodes, desc="Training", initial=episodeIndex) as progressBar:
            while episodeIndex < totalEpisodes:
                episodeIndex += 1
                """环境重置与初始化记录变量"""
                states = env.reset()  # 获取状态
                episodeReturn = 0  # 批次的累计奖励
                doneCount = 0  # 完成的无人机个数（包括成功、碰撞、超过步长、耗尽能量）
                """进行每一步的动作"""
                while doneCount < cfg.uav.uavNums:
                    doneCount = 0  # 清空计数器
                    # 针对于每一个无人机对象
                    for uav in env.unwrapped.uavs:
                        if uav.done:
                            doneCount += 1
                            continue
                        state = states[uav.uavID]  # 获取当前无人机的状态（是一个字典，需要再送入网络中进行处理）
                        action, _ = navigationAlgorithm.take_action(state)  # 算法选择动作
                        nextStates, reward, uavDone, _, information = env.step((action, uav.uavID))  # 与环境交互
                        reward = reward * getattr(cfg, "rewardScale", 0.01)  # 奖励缩放，防止奖励值过大时，网络训练不收敛
                        episodeReturn += reward  # 累计奖励
                        replayBuffer.add(state, action, reward, nextStates, uavDone)  # 放入经验回放池
                        states[uav.uavID] = nextStates  # 更新状态
                """统计无人机的终止状态"""
                # 计数器
                statusCounters = {
                    UAVInfo.SUCCESS: 0,
                    UAVInfo.COLLISION: 0,
                    UAVInfo.POWER_EMPTY: 0,
                    UAVInfo.STEP_OVER: 0,
                }
                for uav in env.unwrapped.uavs:
                    statusCounters[uav.information] += 1
                # 统计
                successCount = statusCounters[UAVInfo.SUCCESS]
                collisionCount = statusCounters[UAVInfo.COLLISION]
                powerExhaustCount = statusCounters[UAVInfo.POWER_EMPTY]
                overCount = statusCounters[UAVInfo.STEP_OVER]
                # 记录奖励
                rewardList.append(episodeReturn)
                """判断是否要升级环境难度"""
                # 如果成功率达到一定的水平，则升级环境难度
                if successCount / getattr(cfg.uav, "uavNums") >= getattr(cfg.env, "successRate", 0.8) and env.unwrapped.level < getattr(cfg.env, "maxLevel", 10):
                    env.unwrapped.level += 1
                    print(f"环境难度已升级为{env.unwrapped.level}")
                """更新网络"""
                # 达到经验回放池最低大小，则更新网络
                if replayBuffer.size() > max(getattr(cfg, "minimalSize", 32), getattr(cfg, "batchSize", 32)):
                    batchSize = getattr(cfg, "batchSize", 32)
                    # 从经验回放池中采样
                    batchStates, batchActions, batchRewards, batchNextStates, batchDones = replayBuffer.sample(batchSize)
                    # 算法更新
                    navigationAlgorithm.update({"states": batchStates,
                                                "actions": batchActions,
                                                "rewards": batchRewards,
                                                "nextStates": batchNextStates,
                                                "dones": batchDones},
                                                env.unwrapped.level)
                    # 首次前向完成时，将神经网络模型的梯度和参数变化注册到wandb平台
                    if wbEnabled and wandb.run is not None and not watchRegistered:
                        try:
                            for name in navigationAlgorithm.checkPointModules:
                                watchTarget = getattr(navigationAlgorithm, name, None)
                                # 由于wandb的限制，只监控 torch.nn.Module (神经网络)，排除优化器(Optimizer)等
                                if watchTarget is not None and isinstance(watchTarget, torch.nn.Module):
                                    wandb.watch(watchTarget, log="all", log_freq=10)
                            watchRegistered = True
                        except Exception as e:
                            print(f"[ERROR] 注册watch失败：{e}")
                else: pass
                """wandb记录指标"""
                if wbEnabled and wandb.run is not None:
                    # 记录指标
                    logData = {
                        "episode": episodeIndex,
                        "episodeReturn": float(episodeReturn),
                        "successCount": int(successCount),
                        "collisionCount": int(collisionCount),
                        "overCount": int(overCount),
                    }
                    # wandb记录指标
                    wandb.log(logData)
                """周期性保存checkPoint，并可上传为artifact"""
                # 保存快照
                if episodeIndex % max(1, saveEvery) == 0:
                    snapshotPath = os.path.join(checkPointDir, f"episode_{episodeIndex}.pt")
                    save_checkPoint(snapshotPath, episodeIndex, navigationAlgorithm)
                    # 上传到wandb
                    upload_wandb(wbEnabled, snapshotPath, f"episode_{episodeIndex}.pt")
                # 保存checkPoint
                if (episodeIndex % max(1, 2) == 0) or (episodeIndex == totalEpisodes):
                    save_checkPoint(lastestCheckPointPath, episodeIndex, navigationAlgorithm, 
                                    extraInfo={"wandb_run_id": wandb.run.id if wbEnabled and wandb.run else None})
                    # 上传到wandb
                    upload_wandb(wbEnabled, lastestCheckPointPath, "latest.pt")
                """输出信息并保存模型"""
                # 输出信息
                progressBar.set_postfix({"当前轮次": episodeIndex,
                                         "环境难度": env.unwrapped.level, 
                                         "奖励": f"{episodeReturn:.2f}", 
                                         "成功": successCount, 
                                         "碰撞": collisionCount, 
                                         "耗尽能量": powerExhaustCount, 
                                         "超过步长": overCount})
                # 更新进度条
                progressBar.update(1)
        """结果绘图"""
        draw_rewards_images(rewardList)
    finally:
        if wbEnabled and wandb.run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()