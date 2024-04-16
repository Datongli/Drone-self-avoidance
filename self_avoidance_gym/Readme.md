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
 
## 使用说明
本项目是基于gymnasium库开发的三维无人机避障训练环境，基于课程学习的思想，由易到难增加环境中建筑物的数量。
整体交互思路如下：
<img src="https://github.com/Datongli/Drone-self-avoidance/blob/194ef0bfb070c29bdffeaad2a2a2462fbc1550d3/self_avoidance_gym/image/%E6%80%BB%E4%BD%93%E7%A4%BA%E6%84%8F%E5%9B%BE.png" alt="交互示意图" width="800" height="400">   
由易到难使用课程学习的思想升级的交互环境如下：
<img src="https://github.com/Datongli/Drone-self-avoidance/blob/194ef0bfb070c29bdffeaad2a2a2462fbc1550d3/self_avoidance_gym/image/%E6%B8%90%E8%BF%9B%E5%BC%8F%E4%BA%A4%E4%BA%92%E7%8E%AF%E5%A2%83.png" alt="课程学习思想升级" width="1000" height="300">  
