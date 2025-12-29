"""
该文件用于编写SAC算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Normal
from navigation.baseNavigationAlgorithm import BaseNavigationAlgorithm


class CircularPositionalEncoding(nn.Module):
    """
    为环形雷达数据设计的位置编码
    基于傅里叶级数的环境位置编码
    """
    def __init__(self, embeddingDim: int=32, maxLength: int=512) -> None:
        """
        雷达位置编码器类初始化函数
        :param embeddingDim: 位置编码维度
        :param maxLength: 最大序列长度
        """
        super().__init__()
        self.embeddingDim = embeddingDim  # 位置编码维度
        """计算每个激光束对应的物理角度 [0, 2π]"""
        # 创建位置索引，第i行的数值就是第i束激光的“角度索引”
        position = torch.arange(maxLength, dtype=torch.float32)  # [maxLength]
        # 角度索引
        angles = position * (2 * math.pi / maxLength)  # [maxLength]
        """创建不同频率的基"""
        numFreqs = embeddingDim // 2
        frequencies = torch.arange(1, numFreqs + 1, dtype=torch.float32)  # [numFreqs]
        """生成编码矩阵[maxLength, embeddingDim]"""
        angleFreq = angles.unsqueeze(1) * frequencies.unsqueeze(0)  # [maxLength, numFreqs]
        # 用sin/cos填充偶数和奇数维
        positionEncoding = torch.zeros(maxLength, embeddingDim)  # [maxLength, embeddingDim]
        # 偶数维度填充sin(k0)，奇数维度填充cos(k0)
        positionEncoding[:, 0::2] = torch.sin(angleFreq)  # [maxLength, embeddingDim / 2]
        positionEncoding[:, 1::2] = torch.cos(angleFreq)  # [maxLength, embeddingDim / 2]
        # 增加一个维度，便于广播操作
        self.register_buffer("positionEncoding", positionEncoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param x: 输入张量，形状为[batchSize, seqLength, embeddingDim]
        :return: 输出张量，形状为[batchSize, seqLength, embeddingDim]
        """
        # 对输入张量的第二维进行位置编码
        x = x + self.positionEncoding[:, :x.size(1)]
        return x
    

class RadarTransformer(nn.Module):
    """
    处理雷达数据的Transformer网络
    """
    def __init__(self, inputDim: int=512, featureDim: int=64, headNum: int=8, layerNum: int=6, outputDim: int=32) -> None:
        """
        处理雷达数据的Transformer网络的初始化函数
        :param inputDim: 输入维度，即雷达数据的维度
        :param featureDim: 特征维度，即Transformer网络的隐藏层维度
        :param headNum: 头数，即Transformer网络中多头注意力机制的头数
        :param layerNum: 层数，即Transformer网络编码器的层数
        :param outputDim: 输出维度，即Transformer网络的输出维度
        """
        super().__init__()
        # 降维层，最终把单束雷达单个距离值升维到featureDim
        self.dimReduction = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, featureDim)
        )
        # 位置编码器，告诉网络每个距离值来自哪个角度
        self.positionEncoder = CircularPositionalEncoding(embeddingDim=featureDim, maxLength=inputDim)
        # Transformer编码器
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=featureDim,
            nhead=headNum, 
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.transformerEncoder = nn.TransformerEncoder(encoderLayer, num_layers=layerNum)
        # 空间注意力增强，对inputDim个位置进行加权求和，生成一个全局特征向量
        # 对每个位置的特征向量进行加权求和，生成一个全局特征向量
        self.spatialAttention = nn.Sequential(
            nn.Linear(featureDim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # 输出层
        self.outputLayer = nn.Sequential(
            nn.Linear(featureDim, featureDim // 2),
            nn.ReLU(),
            nn.Linear(featureDim // 2, outputDim)
        )

    def forward(self, radarData: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param radarData: 输入张量，形状为[batchSize, seqLength, 1]
        :return: 输出张量，形状为[batchSize, featureDim // 4]
        """
        # 重塑为序列
        x = radarData.unsqueeze(-1)  # [batchSize, seqLength, 1]
        # 降维
        x = self.dimReduction(x)  # [batchSize, seqLength, featureDim]
        # 位置编码
        x = self.positionEncoder(x)  # [batchSize, seqLength, featureDim]
        # Transformer编码
        x = self.transformerEncoder(x)  # [batchSize, seqLength, featureDim]
        # 空间注意力增强
        attentionWeights = self.spatialAttention(x)  # [batchSize, seqLength, 1]
        x = torch.sum(x * attentionWeights, dim=1)  # [batchSize, featureDim]
        # 输出层
        x = self.outputLayer(x)  # [batchSize, featureDim // 4]
        return x
    

class StateProcessor(nn.Module):
    """
    处理无人机自身状态
    """
    def __init__(self, inputDim: int=7, hiddenDim: int=64, outputDim: int=32) -> None:
        """
        处理无人机自身状态的初始化函数
        :param inputDim: 输入维度，即无人机状态的维度
        :param hiddenDim: 隐藏层维度，即全连接层的隐藏层维度
        :param outputDim: 输出维度，即处理后的无人机状态的维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim // 2),
            nn.ReLU(),
            nn.Linear(hiddenDim // 2, outputDim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param state: 输入张量，形状为[batchSize, inputDim]
        :return: 输出张量，形状为[batchSize, outputDim]
        """
        return self.net(state)
    

class GatedFusion(nn.Module):
    """
    门控特征融合机制
    """
    def __init__(self, stateDim: int, radarDim: int, fusedDim: int=64) -> None:
        """
        门控特征融合机制的初始化函数
        :param stateDim: 状态维度，即处理后的无人机状态的维度
        :param radarDim: 雷达维度，即处理后的雷达数据的维度
        :param fusedDim: 融合维度，即融合后的特征向量的维度
        """
        super().__init__()
        # 输入维度增加1，用于接收 difficultyLevel
        self.gateInputDim = stateDim + radarDim + 1
        # 状态门
        self.gate = nn.Sequential(
            nn.Linear(self.gateInputDim, fusedDim),
            nn.ReLU(),
            nn.Linear(fusedDim, fusedDim),
            nn.Sigmoid()
        )
        self.stateTransform = nn.Linear(stateDim, fusedDim)
        self.radarTransform = nn.Linear(radarDim, fusedDim)

    def forward(self, stateFeatures: torch.Tensor, radarFeatures: torch.Tensor, difficultyLevel: float) -> tuple:
        """
        前向传播函数
        :param stateFeatures: 状态特征张量，形状为[batchSize, stateDim]
        :param radarFeatures: 雷达特征张量，形状为[batchSize, radarDim]
        :param difficultyLevel: 归一化的困难度，即无人机的困难度，范围为[0, 1]
        :return: 融合后的特征张量，形状为[batchSize, fusedDim]
        """
        batchSize = stateFeatures.size(0)
        # 构造难度张量 [batchSize, 1]
        diffTensor = torch.full((batchSize, 1), difficultyLevel, device=stateFeatures.device)
        # 拼接：状态 + 雷达 + 难度
        gateInput = torch.cat([stateFeatures, radarFeatures, diffTensor], dim=-1)
        # 计算门控值 (0~1)
        # 网络会自动学习：当difficulty低时，gate可能偏向某一侧；高时偏向另一侧
        gate = self.gate(gateInput)
        stateTransformed = self.stateTransform(stateFeatures)
        radarTransformed = self.radarTransform(radarFeatures)
        # 融合
        fused = gate * stateTransformed + (1 - gate) * radarTransformed
        return fused, gate
    

class Actor(nn.Module):
    """
    策略网络
    """
    def __init__(self, stateDim: int=7, radarDim: int=512, actionDim: int=3, actionBound: int=2.0) -> None:
        """
        策略网络的初始化函数
        :param stateDim: 状态维度，即处理后的无人机状态的维度
        :param radarDim: 雷达维度，即处理后的雷达数据的维度
        :param actionDim: 动作维度，即无人机的动作维度
        :param actionBound: 动作边界，即无人机动作的取值范围
        """
        super().__init__()
        self.actionBound = actionBound  # 动作边界限制
        # 特征提取
        self.stateProcessor = StateProcessor(inputDim=stateDim, hiddenDim=64, outputDim=32)
        self.radarTransformer = RadarTransformer(inputDim=radarDim, featureDim=64, outputDim=32)
        # 特征融合
        self.gatedFusion = GatedFusion(stateDim=32, radarDim=32, fusedDim=64)
        # 策略网络
        self.policyNet = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # 输出层
        self.meanLayer = nn.Linear(128, actionDim)
        self.logStdLayer = nn.Linear(128, actionDim)
        # 权重初始化
        self._init_weight()

    def forward(self, stateData: torch.Tensor, radarData: torch.Tensor, difficultyLevel: float) -> tuple:
        """
        前向传播函数
        :param stateData: 状态数据张量，形状为[batchSize, stateDim]
        :param radarData: 雷达数据张量，形状为[batchSize, radarDim]
        :param difficultyLevel: 归一化的困难度，即无人机的困难度，范围为[0, 1]
        :return: 策略张量，形状为[batchSize, actionDim]
        """
        """特征处理"""
        # 特征提取
        stateFeatures = self.stateProcessor(stateData)  # [batchSize, 32]
        radarFeatures = self.radarTransformer(radarData)  # [batchSize, 32]
        # 门控融合
        fusedFeatures, gate = self.gatedFusion(stateFeatures, radarFeatures, difficultyLevel)  # [batchSize, 64]
        """策略网络"""
        x = self.policyNet(fusedFeatures)  # [batchSize, 128]
        # 输出动作分布参数
        mean = self.meanLayer(x)
        logStd = self.logStdLayer(x)
        # 限制logStd的范围
        logStd = torch.tanh(logStd) * 10.0 - 10.0 # 映射到 [-20, 0]
        return mean, logStd, gate

    def get_action(self, stateData: torch.Tensor, radarData: torch.Tensor, deterministic: bool=False, difficultyLevel: float=0.0) -> tuple:
        """
        获取动作函数
        :param stateData: 状态数据张量，形状为[batchSize, stateDim]
        :param radarData: 雷达数据张量，形状为[batchSize, radarDim]
        :param deterministic: 是否确定性策略，默认为False
        :param difficultyLevel: 归一化的困难度，即无人机的困难度，范围为[0, 1]，默认为0.0
        :return: 动作张量，形状为[batchSize, actionDim]
        """
        """获取均值、方差还有门控信号"""
        mean, logStd, gate = self.forward(stateData, radarData, difficultyLevel)
        std = logStd.exp()
        """创建正态分布"""
        normal = Normal(mean, std)
        # print(f"mean: {mean}, std: {std}")
        if deterministic:
            # 确定性策略
            z = mean
        else:
            # 随机策略
            z = normal.rsample()
        """动作绑定"""
        # 计算动作
        action = torch.tanh(z) * self.actionBound
        # 计算logProb
        logProb = normal.log_prob(z) - (2 * (np.log(2) - z - F.softplus(-2 * z))) - np.log(self.actionBound)
        logProb = logProb.sum(dim=-1, keepdim=True)
        return action, logProb, gate
    
    def _init_weight(self) -> None:
        """
        权重初始化函数
        """
        # 对所有线性层和归一化层进行 正交初始化 (Orthogonal Initialization)
        # 这有助于深层网络（如Transformer）的梯度传播
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        # 对均值层和方差层进行 均匀初始化
        self.meanLayer.weight.data.uniform_(-3e-3, 3e-3)
        self.meanLayer.bias.data.uniform_(-3e-3, 3e-3)
        self.logStdLayer.weight.data.uniform_(-3e-3, 3e-3)
        self.logStdLayer.bias.data.fill_(-2.0)
    

class Critic(nn.Module):
    """
    价值网络
    """
    def __init__(self, stateDim: int=7, radarDim: int=512, actionDim: int=3) -> None:
        """
        价值网络的初始化函数
        :param stateDim: 状态维度，即处理后的无人机状态的维度
        :param radarDim: 雷达维度，即处理后的雷达数据的维度
        :param actionDim: 动作维度，即无人机的动作维度
        """
        super().__init__()
        # 共享特征提取
        self.stateProcessor = StateProcessor(inputDim=stateDim, hiddenDim=64, outputDim=32)
        self.radarTransformer = RadarTransformer(inputDim=radarDim, featureDim=64, outputDim=32)
        self.gatedFusion = GatedFusion(stateDim=32, radarDim=32, fusedDim=64)
        # Q1网络（关注短期奖励）
        self.q1Net = nn.Sequential(
            nn.Linear(64 + actionDim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Q2网络（关注长期安全）
        self.q2Net = nn.Sequential(
            nn.Linear(64 + actionDim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 权重初始化
        self._init_weight()

    def forward(self, stateData: torch.Tensor, radarData: torch.Tensor, action: torch.Tensor, difficultyLevel: float) -> torch.Tensor:
        """
        前向传播函数
        :param stateData: 状态数据张量，形状为[batchSize, stateDim]
        :param radarData: 雷达数据张量，形状为[batchSize, radarDim]
        :param action: 动作张量，形状为[batchSize, actionDim]
        :param difficultyLevel: 归一化的困难度，即无人机的困难度，范围为[0, 1]
        :return: Q值张量，形状为[batchSize, 1]
        """
        # 提取特征
        stateFeatures = self.stateProcessor(stateData)
        radarFeatures = self.radarTransformer(radarData)
        # 门控融合
        fusedFeatures, _ = self.gatedFusion(stateFeatures, radarFeatures, difficultyLevel)
        # 拼接动作
        x = torch.cat([fusedFeatures, action], dim=1)
        # 双Q函数
        q1 = self.q1Net(x)
        q2 = self.q2Net(x)
        return q1, q2
    
    def q1(self, stateData: torch.Tensor, radarData: torch.Tensor, action: torch.Tensor, difficultyLevel: float) -> torch.Tensor:
        """
        仅适用Q1网络，用于策略更新
        :param stateData: 状态数据张量，形状为[batchSize, stateDim]
        :param radarData: 雷达数据张量，形状为[batchSize, radarDim]
        :param action: 动作张量，形状为[batchSize, actionDim]
        :param difficultyLevel: 归一化的困难度，即无人机的困难度，范围为[0, 1]
        :return: Q1值张量，形状为[batchSize, 1]
        """
        stateFeatures = self.stateProcessor(stateData)
        radarFeatures = self.radarTransformer(radarData)
        fusedFeatures, _ = self.gatedFusion(stateFeatures, radarFeatures, difficultyLevel)
        x = torch.cat([fusedFeatures, action], dim=-1)
        return self.q1Net(x)

    def _init_weight(self) -> None:
        """
        权重初始化函数
        """
        # 隐藏层通用初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        # 输出层初始化
        self.q1Net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.q1Net[-1].bias.data.uniform_(-3e-3, 3e-3)
        self.q2Net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.q2Net[-1].bias.data.uniform_(-3e-3, 3e-3)


class MTransSAC(BaseNavigationAlgorithm):
    """
    MTransSAC算法实现
    """
    def __init__(self, cfg) -> None:
        """
        算法初始化函数
        :param cfg: 配置对象，包含算法超参数和环境配置
        """
        super(MTransSAC, self).__init__(cfg)
        self.device = getattr(cfg, "device", "cuda:0")  # 设备
        self.actionDim = 3  # 动作维度
        self.actionBound = getattr(cfg, "actionBound", 2)  # 动作范围
        # 策略网络
        self.actor = Actor(
            stateDim=getattr(cfg.uav, "stateDim", 7),
            radarDim=getattr(cfg.uav.sensor, "numBeams", 512),
            actionDim=self.actionDim,
            actionBound=self.actionBound,
        ).to(self.device)
        # 价值网络
        self.critic = Critic(
            stateDim=getattr(cfg.uav, "stateDim", 7),
            radarDim=getattr(cfg.uav.sensor, "numBeams", 512),
            actionDim=self.actionDim
        ).to(self.device)
        # 目标价值网络
        self.targetCritic = Critic(
            stateDim=getattr(cfg.uav, "stateDim", 7),
            radarDim=getattr(cfg.uav.sensor, "numBeams", 512),
            actionDim=self.actionDim
        ).to(self.device)
        # 先使用一个固定的Alpha值
        self.fixedAlpha = 0.0001
        # 复制目标网络参数
        self.targetCritic.load_state_dict(self.critic.state_dict())
        # 自动调节温度参数
        # self.targetEntropy = -torch.prod(torch.Tensor([self.actionDim])).item()  # 目标熵
        self.targetEntropy = -2.0
        # self.logAlpha = torch.zeros(1, requires_grad=True, device=self.device)  # 温度参数Alpha，越大越喜欢探索
        self.logAlpha = torch.tensor(np.log(0.01), requires_grad=True, device=self.device) 
        if getattr(cfg, "mode", "train") == "train":
            self.alphaOptimizer = torch.optim.Adam([self.logAlpha], lr=getattr(cfg, "learningRateAlpha", 1e-4))  # alpha优化器
            # 优化器
            self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=getattr(cfg, "learningRateActor", 1e-3))  # Actor网络优化器
            self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=getattr(cfg, "learningRateCritic", 1e-3))  # Critic网络优化器
            self.actorScheduler = torch.optim.lr_scheduler.StepLR(self.actorOptimizer,
                                                                        step_size=getattr(cfg, "actorLrStepSize", 500), 
                                                                        gamma=getattr(cfg, "actorLrDecay", 0.9))  # Actor网络学习率衰减器
            self.criticScheduler = torch.optim.lr_scheduler.StepLR(self.criticOptimizer, 
                                                                        step_size=getattr(cfg, "criticLrStepSize", 500), 
                                                                        gamma=getattr(cfg, "criticLrDecay", 0.9))  # Critic网络学习率衰减器
        else:
            # 测试模式下，将优化器设置为 None，防止后续代码误调用报错
            self.alphaOptimizer = None
            self.actorOptimizer = None
            self.criticOptimizer = None
            self.actorScheduler = None
            self.criticScheduler = None
        # 课程学习参数
        self.difficultyLevel = 1  # 课程难度
        self.updateInterval = 0  # 课程更新间隔
        # 需要进行参数加载的模块
        self.checkPointModules = ["actor", "critic", "targetCritic", "actorOptimizer", "criticOptimizer",
                                  "actorScheduler", "criticScheduler", "logAlpha", "alphaOptimizer",
                                  "difficultyLevel", "updateInterval"]

    def take_action(self, stateDict: dict, deterministic: bool = False) -> tuple:
        """
        选择动作函数(支持 Batch Inference)
        :param stateDict: 状态字典，包含状态数据和雷达数据
        :param deterministic: 是否确定性选择动作，默认为False
        :return: 动作数组 (numpy array), 门控数组 (numpy array)
        """
        # 提取无人机状态数据
        uavState = torch.FloatTensor(stateDict["uavState"]).to(self.device)
        # 提取雷达数据
        radarData = torch.FloatTensor(stateDict["sensorState"]).to(self.device)
        # 兼容一个无人机输入的情况
        if uavState.dim() == 1:
            uavState = uavState.unsqueeze(0)
        if radarData.dim() == 1:
            radarData = radarData.unsqueeze(0)
        # 获得归一化的难度等级
        normalizedDifficulty = self._get_normalized_difficulty()
        # 获取动作
        with torch.no_grad():
            action, _, gate = self.actor.get_action(uavState, radarData, deterministic, normalizedDifficulty)
        # 转换为numpy数组
        # 保留 [BatchSize, ActionDim] 的形状，方便外部索引
        action = action.cpu().numpy()
        gate = gate.cpu().numpy()
        return action, gate
    
    def update(self, batchData: dict, envLevel: int) -> dict:
        """
        更新算法参数
        :param batchData: 批量数据，包含状态数据、动作数据、奖励数据、下一个状态数据、是否终止数据
        :param envLevel: 环境难度级别
        :return: 更新信息字典
        """
        self.updateInterval += 1  # 课程更新间隔加1
        self.difficultyLevel = envLevel  # 更新课程难度
        normalizedDifficulty = self._get_normalized_difficulty()  # 获取归一化的难度等级
        """转换数据到tensor"""
        uavStates = torch.FloatTensor(np.array(batchData['states']['uavState'])).to(self.device)
        radarData = torch.FloatTensor(np.array(batchData['states']['sensorState'])).to(self.device)
        actions = torch.FloatTensor(np.array(batchData['actions'])).to(self.device)
        rewards = torch.FloatTensor(np.array(batchData['rewards'])).unsqueeze(1).to(self.device)
        nextUavStates = torch.FloatTensor(np.array(batchData['nextStates']['uavState'])).to(self.device)
        nextRadarData = torch.FloatTensor(np.array(batchData['nextStates']['sensorState'])).to(self.device)
        dones = torch.FloatTensor(np.array(batchData['dones'])).unsqueeze(1).to(self.device)
        """更新Critic网络"""
        with torch.no_grad():
            nextActions, nextLogProb, _ = self.actor.get_action(nextUavStates, nextRadarData, normalizedDifficulty)
            nextQ1, nextQ2 = self.targetCritic(nextUavStates, nextRadarData, nextActions, normalizedDifficulty)
            nextQ = torch.min(nextQ1, nextQ2) - torch.exp(self.logAlpha) * nextLogProb
            # nextQ = torch.min(nextQ1, nextQ2) - self.fixedAlpha * nextLogProb
            targetQ = rewards + (1 - dones) * getattr(self.cfg, "gamma", 0.99) * nextQ
        # 当前Q值
        currentQ1, currentQ2 = self.critic(uavStates, radarData, actions, normalizedDifficulty)
        # 计算Critic损失
        criticLoss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
        # 回传损失
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), getattr(self.cfg, "maxGradNorm", 1.0))
        self.criticOptimizer.step()
        """更新Actor网络"""
        newActions, logProb, gate = self.actor.get_action(uavStates, radarData, normalizedDifficulty)
        Q1new = self.critic.q1(uavStates, radarData, newActions, normalizedDifficulty)  # 只计算Q1，是常见的加速技巧
        # 计算专家动作
        expertActions = self._calculate_expert_action(uavStates)
        # 课程学习：根据难度级别调整熵权重
        alpha = self._get_alpha()
        sacLoss = (alpha * logProb - Q1new).mean()  # SAC损失
        # actorLoss = (self.fixedAlpha * logProb - Q1new).mean()
        bcLoss = F.mse_loss(newActions, expertActions)  # 行为克隆损失
        expertWeight = max(0.1, 1.0 - (self.difficultyLevel - 1) * 0.1)  # 动态权重
        actorLoss = expertWeight * bcLoss + (1 - expertWeight) * sacLoss  # 策略损失
        # 回传损失
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), getattr(self.cfg, "maxGradNorm", 1.0))
        self.actorOptimizer.step()
        """更新温度参数alpha"""
        alphaLoss = -(self.logAlpha * (logProb + self.targetEntropy).detach()).mean()
        self.alphaOptimizer.zero_grad()
        alphaLoss.backward()
        self.alphaOptimizer.step()
        """软更新目标网络"""
        if self.updateInterval % getattr(self.cfg, "targetUpdateInterval", 2) == 0:
            self._soft_update_target_network()
        """更新学习率"""
        # 更新actor网络学习率
        self.actorScheduler.step()
        # 更新critic网络学习率
        self.criticScheduler.step()
        return {
            'criticLoss': criticLoss.item(),
            'actorLoss': actorLoss.item(),
            'bcLoss': bcLoss.item(),
            'expertWeight': expertWeight,
            'alphaLoss': alphaLoss.item(),
            'alpha': alpha.item(),
            # 'alpha': self.fixedAlpha,
            'gateMean': gate.mean().item(),
            'q1Mean': currentQ1.mean().item(), 
            'q2Mean': currentQ2.mean().item()
        }
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        :param path: 模型保存路径
        :return: None
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOptimizer': self.actorOptimizer.state_dict(),
            'criticOptimizer': self.criticOptimizer.state_dict(),
            'logAlpha': self.logAlpha,
            'alphaOptimizer': self.alphaOptim.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        """
        加载模型    
        :param path: 模型加载路径
        :return: None
        """
        # 取出检查点
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.targetCritic.load_state_dict(checkpoint['targetCritic'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.logAlpha = checkpoint['logAlpha']
        self.alphaOptim.load_state_dict(checkpoint['alphaOptimizer'])

    def _get_normalized_difficulty(self) -> float:
        """
        获取归一化的难度等级
        """
        maxLevel = getattr(self.cfg, "maxLevel", 10)
        return min(self.difficultyLevel / maxLevel, 1.0)
    
    def _calculate_expert_action(self, uavStates: torch.Tensor) -> torch.Tensor:
        """
        专家系统：计算指向目标点的最佳动作
        :param uavStates: 无人机状态张量 [batchSize, stateDim]
        :return: 专家动作张量 [batchSize, actionDim]
        """
        # 提取归一化的相对坐标
        normDelta = uavStates[:, :3]
        """反归一化"""
        length = getattr(self.cfg.env, "length", 100)
        width = getattr(self.cfg.env, "width", 100)
        height = getattr(self.cfg.env, "height", 25)
        scale = torch.tensor([length, width, height], device=self.device)  # 创建缩放张量
        realDelta = normDelta * scale  # 真实的物理位移偏移量
        """计算专家动作"""
        # 计算单位方向向量
        dist = torch.norm(realDelta, p=2, dim=1, keepdim=True)
        direction = realDelta / (dist + 1e-6)
        # 映射到动作空间
        expertAction = direction * getattr(self.cfg, "actionBound", 2.0)
        return expertAction

    def _get_alpha(self) -> float:
        """
        获取温度参数Alpha
        :return: 温度参数Alpha
        """
        # 获得基准熵
        baseAlpha = self.logAlpha.exp()
        # 难度级别越高，探索权重越低
        difficultyFactor = 1.0 / (1 + 0.2 * (self.difficultyLevel - 1))
        return baseAlpha * difficultyFactor
    
    def _soft_update_target_network(self, tau: float = 0.005) -> None:
        """
        软更新目标网络
        :param tau: 软更新参数，默认为0.005
        """
        for targetParam, param in zip(self.targetCritic.parameters(), self.critic.parameters()):
            targetParam.data.copy_(tau * param.data + (1 - tau) * targetParam.data)

            



    




if __name__ == "__main__":
    pass