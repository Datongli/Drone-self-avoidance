"""
该文件用于构建坐标类
"""
from dataclasses import dataclass


@dataclass
class Coordinate:
    """坐标类"""
    x: int | float = 0  # x坐标
    y: int | float = 0  # y坐标
    z: int | float = 0  # z坐标


if __name__ == '__main__':
    pass