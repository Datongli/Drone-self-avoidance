# 说明文档
## 文件结构
- self_avoidance_gym
  - environment-gym
    - environment_gym
      - envs
        - \__init__.py
        - uav_avoid.py
      - wrapppers
        - \__init__.py
      - \__init__.py
    - setup.py
  - pth
  - Readme.md
  - DDPG.py
  - SAC.py
  - main.py
  - test_uav.py
  - tools.py
 
## 项目说明
本项目是**基于gymnasium库**开发的**连续三维无人机避障训练环境**，基于课程学习的思想，由易到难增加环境中建筑物的数量。
本项目编写的环境状态连续，动作空间连续，可用于训练深度强化学习中的**连续算法**。
整体交互思路如下：
<img src="https://github.com/Datongli/Drone-self-avoidance/blob/194ef0bfb070c29bdffeaad2a2a2462fbc1550d3/self_avoidance_gym/image/%E6%80%BB%E4%BD%93%E7%A4%BA%E6%84%8F%E5%9B%BE.png" alt="交互示意图" width="800" height="400">   
由易到难使用课程学习的思想升级的交互环境如下：
<img src="https://github.com/Datongli/Drone-self-avoidance/blob/194ef0bfb070c29bdffeaad2a2a2462fbc1550d3/self_avoidance_gym/image/%E6%B8%90%E8%BF%9B%E5%BC%8F%E4%BA%A4%E4%BA%92%E7%8E%AF%E5%A2%83.png" alt="课程学习思想升级" width="1000" height="300">  
基于gymnasium的环境，本项目封装在./environment-gym/environment_gym/envs/uav_avoid.py中，其中实现了诸如reset(),step(),render()等符合gymnasium规范的函数

## 示例文件说明
- self_acoidance_gym
  - DDPG.py
    - DDPG算法实现的文件
    - 其中将避障和寻迹网络分开
      - 避障网络：卷积网络
      - 寻迹网络：全连接网络
    - 增加了专家系统的参与，计算loss，避免初期无意义地振荡
  - SAC.py
    - SAC算法实现文件
    - 本项目**没有调试完成**，仅供后来者参考
  - main.py
    - 初始化交互环境对象
    - 设定一些训练参数
  - test_uav.py
    - 验证算法性能的文件
  - tool.py
    - 定义了一些工具
    - 离线策略训练主体过程

## 初始定义和软件包依赖
在./environment-gym/environment_gym/\__init__.py中，给出了项目对于环境的初始定义和声明：
```python
from gymnasium.envs.registration import register
import numpy as np
from environment_gym.envs.uav_avoid import UavAvoidEnv

register(
    id='environment_gym/UavAvoid-v0',
    entry_point="environment_gym.envs:UavAvoidEnv",
    kwargs={
        "agent_r": 1.0,
        "action_area": np.array([[0, 0, 0], [100, 100, 25]]),
        "action_bound": 2.0,
        "uavs_num": 15,
        "render_mode": None
    },
)
```
在./environment-gym/setup.py中，给出了项目的版本号和需要满足的其他软件包依赖，如果**安装时引发依赖冲突请使用者自查**
```python
from setuptools import setup

setup(
    name="uav_avoid_gym",
    version="0.0.1",
    install_requires=["gymnasium==0.29.1", "numpy==1.24.1", "matplotlib==3.8.2"],
)
```

## 安装
本项目是基于gymnasium包编写，使用者应了解其规范和使用方式
（若未安装gymnasium，在安装本项目时setup.py会解决相关依赖，安装gymnasium）
**安装本项目流程**如下：
- 将environment-gym文件夹打包下载到本地
- 将environment-gym文件夹的绝对路径传入，例如
  ```shell
  pip install D:/project/environment-gym
  ```
- 显示成功安装
  ```shell
  Successfully installed uav-avoid-gym-0.0.1
  ```
安装及成功安装示意图如下：
<img src="https://github.com/Datongli/Drone-self-avoidance/blob/80a541971608e810385686f6885a8c62ffcd7e6f/self_avoidance_gym/image/%E5%AE%89%E8%A3%85%E9%A1%B9%E7%9B%AE%E4%B8%8E%E6%88%90%E5%8A%9F.png" alt="安装与成功" width="800" height="300">  

使用pip install -e 安装是项目在本地调试阶段时，为了使项目更改立马作用的选择，使用者**请忽略 -e的修饰**，直接使用pip install安装即可

## 使用示例
在main.py中，可以如此使用：
```python
import gymnasium
import environment_gym
import numpy as np

agent_r = 1.0  # 无人机对象的半径
action_area = np.array([[0, 0, 0],[100, 100, 25]])  # 避障空间的限制，xyz，第一行是最小值，第二行是最大值
action_boung = 2.0  # 无人机一步最多能走的限制
uav_num = 15  # 一个批次训练无人机的数量
render_mode = None  # 观测模式
env = gymnasium.make('environment_gym/UavAvoid-v0',
                         agent_r=agent_r,
                         action_area=action_area,
                         action_bound=action_bound,
                         uavs_num=uavs_num,
                         render_mode=render_mode)
for _ in range(1000):  # 假设训练1000次
  state, _ = env.reset()  # 重置环境，得到起始状态
  for i in range(uav_num):
    …………
    # action根据训练的算法给出，i为第i个无人机
    next_state, reward, uav_done, _, Info = env.step((action[0], i))  # 根据选取的动作改变状态，获取收益
    …………

```
其中render_mode可以选择None和"human"两种模式，但是由于作者水平有限，在训练阶段多个无人机一起显示会引发matplotlib绘图程序崩溃，因此训练时建议使用者采用None模式，在验证算法效果时使用1个无人机，采用"human"模式

**也可以**通过下面的方式实例化环境：
```python
import environment_gym

agent_r = 1.0  # 无人机对象的半径
action_area = np.array([[0, 0, 0],[100, 100, 25]])  # 避障空间的限制，xyz，第一行是最小值，第二行是最大值
action_boung = 2.0  # 无人机一步最多能走的限制
uav_num = 15  # 一个批次训练无人机的数量
render_mode = "human"  # 观测模式
env = environment_gym.UavAvoidEnv(agent_r=agent_r,
                                  action_area=action_area,
                                  action_bound=action_bound,
                                  uavs_num=uavs_num,
                                  render_mode=render_mode)
```
## 使用展示
<img src="https://github.com/Datongli/Drone-self-avoidance/blob/cc2f3bcaac7826227a485f12fc914a4530745c27/self_avoidance_gym/image/%E4%BD%BF%E7%94%A8%E5%B1%95%E7%A4%BA.png" alt="使用展示" width="1000" height="500">
