"""
建筑物
"""
from dataclasses import dataclass
from ..environment.coordinate import Coordinate


@dataclass
class Building:
    """建筑物类"""
    x: int | float = 0  # 建筑物中心的x坐标
    y: int | float = 0  # 建筑物中心的y坐标
    halfX: int | float = 0  # 建筑物x方向长度的一半
    halfY: int | float = 0  # 建筑物y方向长度的一半
    height: int | float = 0  # 建筑物的高度
    leftDown: Coordinate = None  # 建筑物左下角的坐标
    rightUp: Coordinate = None  # 建筑物右上角的坐标


if __name__ == '__main__':
    pass
