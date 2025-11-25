"""
该文件用于编写SAC算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Normal


class CircularPositionalEncoding(nn.Module):
    """为环形雷达数据设计的位置编码"""
    def __init__(self, embeddingDim: int=32, maxLength: int=512) -> None:
        """
        雷达位置编码器类初始化函数
        :param embeddingDim: 位置编码维度
        :param maxLength: 最大序列长度
        """
        super().__init__()
        self.embeddingDim = embeddingDim  # 位置编码维度
        """创建环形位置编码(0度和360度连接起来看待)"""
        # 创建位置索引，第i行的数值就是第i束激光的“角度索引”
        position = torch.arange(maxLength).unsqueeze(1)  # [maxLength, 1]
        # 创建频率因子，用于计算位置编码的sin和cos值，给不同维度分配不同的频率
        frequencyFactor = torch.exp(
            torch.arange(0, embeddingDim, 2) * (-math.log(10000.0) / embeddingDim)
        )  # [embeddingDim / 2]
        # 创建位置编码矩阵
        positionEncoding = torch.zeros(maxLength, embeddingDim)  # [maxLength, embeddingDim]
        # 用sin/cos填充偶数和奇数维
        # 同一个角度，会得到一个长维embeddingDim维的向量
        positionEncoding[:, 0::2] = torch.sin(position * frequencyFactor * 2 * math.pi / maxLength)  # [maxLength, embeddingDim / 2]
        positionEncoding[:, 1::2] = torch.cos(position * frequencyFactor * 2 * math.pi / maxLength)  # [maxLength, embeddingDim / 2]
        """增强首尾衔接"""
        smoothingFactor = 0.3  # 



if __name__ == "__main__":
    pass