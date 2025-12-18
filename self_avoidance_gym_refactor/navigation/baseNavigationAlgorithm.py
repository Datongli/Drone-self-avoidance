class BaseNavigationAlgorithm:
    """
    导航算法基类
    """

    def __init__(self, cfg) -> None:
        """
        算法初始化函数
        :param cfg: 配置文件
        """
        self.cfg = cfg

    def take_action(self) -> None:
        """
        执行动作函数
        """
        pass

    def update(self) -> None:
        """
        更新函数
        """
        pass

    def save_model(self) -> None:
        """
        保存模型函数
        """
        pass

    def load_model(self) -> None:
        """
        加载模型函数
        """
        pass


if __name__ == "__main__":
    pass