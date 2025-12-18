"""
工具函数文件
"""
import collections
import numpy as np
import random
import wandb
import os
import torch
import matplotlib.pyplot as plt
# 可以显示中文
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity: int) -> None:
        # 一个先进先出的队列
        # 创建了一个具有固定容量（capacity）的双端队列（deque）
        # maxlen 参数指定了经验缓冲区的最大容量，即队列中元素的最大数量。
        # 当缓冲区达到最大容量时，添加新元素会导致旧元素从队列的另一端被移除，以保持缓冲区不超过设定的最大值。
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: dict, action: np.array, reward: float, nextState: dict, done: bool) -> tuple:
        """
        将数据加入buffer中
        :param state: 状态
        :param action: 动作
        :param reword: 奖励
        :param next_state:动作之后的下一状态
        :param done:环境是否结束
        :return: 更新容器
        """
        self.buffer.append((state.copy(), action.copy(), reward, nextState.copy(), done))

    def sample(self, batchSize: int) -> tuple:
        """
        从缓冲区挑选数量为batchSize的样本
        :param batchSize: 要挑选的数量
        :return: 状态，动作，奖励，下一状态，done
        """
        # 从经验回放缓冲区中无放回地随机选择batch_size个样本（但是选择后缓冲区中经验数量不变）
        transitions = random.sample(self.buffer, batchSize)
        # 将 transitions 中的元素按列解压缩成五个列表
        # 并分别赋值给 state、action、reward、next_state 和 done
        state, action, reward, nextState, done = zip(*transitions)
        # 由于state和next_state都是字典，因此需要再经过一些特殊的处理
        batchState = {
            'uavState': np.array([s['uavState'] for s in state]),
            'sensorState': np.array([s['sensorState'] for s in state])
        }
        batchNextState = {
            'uavState': np.array([s['uavState'] for s in nextState]),
            'sensorState': np.array([s['sensorState'] for s in nextState])
        }
        return batchState, action, reward, batchNextState, done

    def size(self) -> int:
        """
        读取目前缓冲区中数据的数量
        :return: 缓冲区中数据数量
        """
        return len(self.buffer)


def moving_average(a, windowSize: int = 50):
    """
    计算移动平均值
    :param a: 输入数组
    :param windowSize: 窗口大小，默认值为50
    :return: 移动平均值数组
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[windowSize:] - cumulative_sum[:-windowSize]) / windowSize
    r = np.arange(1, windowSize - 1, 2)
    begin = np.cumsum(a[:windowSize - 1])[::2] / r
    end = (np.cumsum(a[:-windowSize:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def draw_rewards_images(rewardList: list) -> None:
    """
    绘制奖励图像
    :param rewardList: 奖励列表
    :return: None
    """
    episodesList = list(range(len(rewardList)))
    mvReturn = moving_average(rewardList, 9)
    # 创建一个二维图形窗口
    fig2d = plt.figure()
    # 添加一个二维子图
    ax2d = fig2d.add_subplot(1, 1, 1)
    ax2d.plot(episodesList, rewardList, color='blue', label='原本训练记录')
    ax2d.plot(episodesList, mvReturn, color='orange', label='滑动平均记录')
    ax2d.legend()
    ax2d.set_xlabel('训练轮次')
    ax2d.set_ylabel('返回值')
    ax2d.set_title('综合')
    # 显示图像
    plt.show()


def cfg_get(cfg, dottedKey: str, default: any)-> any:
    """
    安全读取配置文件中的参数
    支持用点分隔的多级键
    若任一层不存在或者访问失败，返回default，避免KeyError/AttributeError
    :param cfg: 配置文件对象
    :param dottedKey: 键路径
    :param default: 默认值
    :return: 参数值
    """
    try:
        cursor = cfg  # 游标
        # 遍历键路径中的每个键
        for key in dottedKey.split('.'):
            cursor = getattr(cursor, key)
        return cursor
    except Exception:
        # 若任意层不存在或者访问失败，返回默认值
        return default
    

def to_plain_cfg(cfg) -> dict[str, any]:
    """
    将 OmegaConf 配置对象转换为可序列化的 Python dict（resolve 插值）
    在无法导入omegaconf或转换异常时返回空字典，避免wandb.init崩溃
    :param cfg: 配置文件对象
    :return: 可序列化的 Python dict
    """
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return {}
    

def wandb_init(cfg, upDir: str) -> None:
    """
    初始化wandb运行
    :param cfg: 配置文件对象
    :param upDir: 项目根目录
    :return: 
    """
    wbEnabled = bool(cfg_get(cfg, "wandb.enabled", True))  # 是否使用wandb
    wbProject = cfg_get(cfg, "wandb.project", "XTDrone-DRL")  # wandb项目名称
    wbEntity = cfg_get(cfg, "wandb.entity", "XTDrone-DRL-UAV-Navigation")  # wandb实体名称
    wbName = cfg_get(cfg, "wandb.name", "navigation-training-gym")  # wandb实验名称
    wbMode = cfg_get(cfg, "wandb.mode", "online")  # wandb模式
    wbDirCfg = cfg_get(cfg, "wandb.dir", "wandb")  # wandb保存目录
    wbDir = os.path.join(upDir, wbDirCfg)
    os.makedirs(wbDir, exist_ok=True)  # 创建wandb保存目录
    wbId = cfg_get(cfg, "wandb.id", None)  # wandb实验ID
    wbResumeFlag = cfg_get(cfg, "wandb.resumeFlag", True)  # 是否从上次断点继续
    # 初始化wandb运行
    if wbEnabled:
        wandb.init(
            project=wbProject,
            entity=wbEntity,
            name=wbName,
            mode=wbMode,
            id=wbId if wbResumeFlag and wbId else None,  # 指定id以便合并到到同一个run
            resume="allow" if wbResumeFlag and wbId else None,  # 仅在提供id时允许自动续接
            dir=wbDir,
            config=to_plain_cfg(cfg),  # 将完整Hydra配置同步到run.config
        )


def load_checkPoint(checkPointPath: str, navigationAlgorithm: any) -> int:
    """
    加载检查点
    :param checkPointPath: 检查点路径
    :param navigationAlgorithm: 导航算法对象
    :return: 加载的episode数目
    """
    """加载检查点"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有可用的设备
    checkPointState: dict[str, any] = torch.load(checkPointPath, map_location=device, weights_only=False)  # 加载检查点状态
    navigationState: dict[str, any] = checkPointState.get("navigartionAlgorithm", {})  # 获取导航算法状态
    """加载导航算法状态（支持注册表自动化加载）"""
    try:
        modulesToLoad = getattr(navigationAlgorithm, "checkPointModules", ["actor", "critic", "targetCritic", "actorOptimizer", "criticOptimizer",
                                  "actorScheduler", "criticScheduler", "logAlpha", "alphaOptimizer",
                                  "difficultyLevel", "updateInterval"])  # 获取需要加载检查点的模块列表
        for moduleName in modulesToLoad:
            # 从算法对象中获取对应的属性
            if not hasattr(navigationAlgorithm, moduleName):
                print(f"导航算法对象中不存在属性 {moduleName}，跳过加载")
                continue  # 若属性不存在，跳过
            targetAttr = getattr(navigationAlgorithm, moduleName)  # 获取目标属性
            savedValue = navigationState.get(moduleName, None)  # 获取保存的属性值
            if savedValue is None:
                print(f"导航算法状态中不存在属性 {moduleName}，跳过加载")
                continue  # 若属性不存在，跳过
            # 如果是模型对象或者优化器，加载参数
            if hasattr(targetAttr, "load_state_dict"):
                safe_load_state(targetAttr, savedValue)
            # 如果是普通属性，加载属性值
            else:
                setattr(navigationAlgorithm, moduleName, savedValue)
    except Exception as e:
        print(f"加载检查点时获取导航算法状态失败：{e}")
        return
    """加载检查点中的episode数目"""
    try:
        episode: int = checkPointState.get("episode", 0)
        return episode
    except Exception as e:
        print(f"加载episode数目时失败：{e}")
        return 0 
    

def safe_load_state(targetObject: torch.nn.Module | None, stateDict: dict | None) -> None:
    """
    安全加载模型的状态字典（state_dict）
    若模型未定义加载状态字典方法或加载过程中发生异常，忽略，避免崩溃
    :param targetObject: 模型对象
    :param stateDict: 状态字典
    :return: None
    """
    if targetObject is not None and hasattr(targetObject, "load_state_dict") and stateDict is not None:
        try:
            targetObject.load_state_dict(stateDict)
        except Exception:
            pass


def save_checkPoint(checkPointPath: str, episodeIndex: int, navigationAlgorithm: any, extraInfo: dict[str, any] = None) -> None:
    """
    保存检查点
    :param checkPointPath: 检查点路径
    :param episodeIndex: 当前episode数目
    :param navigationAlgorithm: 导航算法对象
    :param extraInfo: 额外信息字典
    :return: None
    """
    # 确保检查点目录存在
    os.makedirs(os.path.dirname(checkPointPath), exist_ok=True)
    # 准备保存的状态字典
    checkPointState = {
        "episode": episodeIndex,  # 当前episode数目
        "navigartionAlgorithm": {},  # 导航算法状态
        "extraInfo": extraInfo,  # 额外的信息
    }
    """保存导航算法状态（支持注册表自动化保存）"""
    try:
        modulesToSave = getattr(navigationAlgorithm, "checkPointModules", ["actor", "critic", "targetCritic", "actorOptimizer", "criticOptimizer",
                                  "actorScheduler", "criticScheduler", "logAlpha", "alphaOptimizer",
                                  "difficultyLevel", "updateInterval"])  # 获取需要保存检查点的模块列表
        for moduleName in modulesToSave:
            # 从算法对象中获取对应的属性
            if not hasattr(navigationAlgorithm, moduleName):
                print(f"导航算法对象中不存在属性 {moduleName}，跳过保存")
                continue  # 若属性不存在，跳过
            savedValue: any  = getattr(navigationAlgorithm, moduleName)  # 获取属性值
            # 如果是模型对象或者优化器，保存参数
            if hasattr(savedValue, "state_dict"):
                checkPointState["navigartionAlgorithm"][moduleName] = savedValue.state_dict()
            # 如果是普通属性，保存属性值
            else:
                checkPointState["navigartionAlgorithm"][moduleName] = savedValue
    except Exception as e:
        print(f"获取导航算法状态时发生错误：{e}")
    try:
        torch.save(checkPointState, checkPointPath)
    except Exception as e:
        print(f"保存检查点时发生错误：{e}")


def upload_wandb(wbEnabled: bool, path: str, name: str) -> None:
    """
    上传wandb artifact
    :param wbEnabled: 是否启用wandb
    :param path: artifact路径
    :param name: artifact名称
    :return: None
    """
    if wbEnabled and wandb.run is not None:
        try:
            artifact = wandb.Artifact("sac-checkPoints", type="model")
            artifact.add_file(path, name=name)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"上传wandb artifact时发生错误：{e}")
    else:
        return 
    

def switch_model_eval(navigationAlgorithm: any) -> None:
    """
    将导航算法切换为eval模式
    :param navigationAlgorithm: 导航算法对象
    :return: None
    """
    if not hasattr(navigationAlgorithm, "checkPointModules"):
        return
    try:
        modelsToSwitch = getattr(navigationAlgorithm, "checkPointModules", ["actor", "critic", "targetCritic"])
        for model in modelsToSwitch:
            # 只监控有定义且是可调用对象的模块
            if hasattr(model, "eval") and callable(getattr(model, "eval")):
                model.eval()
            else: pass
    except Exception as e:
        print(f"[ERROR] 模型切换eval模式失败：{e}")



if __name__ == '__main__':
    pass