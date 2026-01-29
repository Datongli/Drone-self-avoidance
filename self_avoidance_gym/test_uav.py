"""
用于进行生成实际环境，并对保存的参数进行测试的文件
"""
import environment_gym
import matplotlib
import os
from pathlib import Path
from DDPG import *
from SAC import *


# 以当前脚本所在目录作为基准
BASE_DIR = Path(__file__).resolve().parent
PTH_DIR = BASE_DIR / "pth"
matplotlib.use('TkAgg')  # 或者其他后端
# 选择模型
test_model = 'DDPG'
# 策略网络学习率
actor_lr = 1e-3
# 价值网络学习率
critic_lr = 1e-3
# 迭代次数
num_episodes = 50000
# 隐藏节点，先暂定64，后续可以看看效果
hidden_dim = 64
# 折扣因子
gamma = 0.99
# 软更新参数
tau = 0.05
# 经验回放池大小
buffer_size = 10000
# 每一批次选取的经验数量
batch_size = 128
# 经验回放池最小经验数目
minimal_size = batch_size
# 高斯噪声标准差
sigma = 0.01
# 三维环境下动作，加上一堆状态的感知，目前是124+16=140个
state_dim = 21
# 暂定直接控制智能体的位移，所以是三维的
action_dim = 3
# 无人机可控风速
v0 = 40
# 设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 智能体的半径（先暂时定义为球体）
agent_r = 0.2
# 动作区域
action_area = np.array([[0, 0, 0], [100, 100, 25]])
# 动作最大值
action_bound = 2.0
# 目标熵，用于SAC算法
target_entropy = - action_dim
# SAC模型中的alpha参数学习率
alpha_lr = 1e-5
# 最大贪心次数
max_eps_episode = 0
# 最小贪心概率
min_eps = 0
wd = 0.0
# 测试次数
n_test = 1
# 无人机数量
uavs_num = 1
# 环境等级
level = 15
# 渲染模式
render_mode = "human" if n_test == 1 and uavs_num == 1 else None
# 随机数种子
seed = 131 if n_test == 1 and uavs_num == 1 else None


if __name__ == '__main__':
    if test_model == 'DDPG':
        agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False,
                     action_bound, sigma, actor_lr, critic_lr, tau, gamma, max_eps_episode, min_eps, wd, device)

        pth_load = {
            'actor': str(PTH_DIR / 'actor.pth'),
            'critic': str(PTH_DIR / 'critic.pth'),
            'target_actor': str(PTH_DIR / 'target_actor.pth'),
            'target_critic': str(PTH_DIR / 'target_critic.pth'),
        }

    if test_model == 'SAC':
        agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr,
                              alpha_lr, target_entropy, tau, gamma, max_eps_episode, min_eps, wd, device)
        pth_load = {
            'SAC_actor': str(PTH_DIR / 'SAC_actor.pth'),
            "critic_1": str(PTH_DIR / 'critic_1.pth'),
            "critic_2": str(PTH_DIR / 'critic_2.pth'),
            'target_critic_1': str(PTH_DIR / 'target_critic_1.pth'),
            'target_critic_2': str(PTH_DIR / 'target_critic_2.pth'),
        }
    # env = gymnasium.make('environment_gym/UavAvoid-v0',
    #                      agent_r=agent_r,
    #                      action_area=action_area,
    #                      action_bound=action_bound,
    #                      uavs_num=uavs_num,
    #                      render_mode="human")
    env = environment_gym.UavAvoidEnv(agent_r=agent_r, action_area=action_area,
                                      action_bound=action_bound, uavs_num=uavs_num, render_mode=render_mode)
    for name, pth in pth_load.items():
        # 按照键名称取出存档点
        check_point = torch.load(pth_load[name], map_location=device)
        # 装载模型参数
        agent.net_dict[name].load_state_dict(check_point['model'])
    # 真实场景运行
    env.unwrapped.level = level  # 环境难度等级
    env.unwrapped.uavs_num = uavs_num  
    total_reward = 0
    n_success = 0
    count = 0
    n_creash = 0  # 坠毁数目

    for _ in range(n_test):
        states, _ = env.reset(seed=seed)  # 环境重置
        n_done = 0
        # 保险：避免极端情况下策略/环境导致永不终止
        max_total_steps = 200000
        total_steps = 0
        while True:
            total_steps += 1
            if total_steps > max_total_steps:
                print(f"[WARN] 超过最大步数 {max_total_steps}，强制结束本次测试，避免死循环。")
                break
            for i in range(len(env.unwrapped.uavs)):
                if env.uavs[i].done:
                    # 无人机已结束任务，跳过
                    continue
                # 更新agent中的步数
                agent.step = env.uavs[i].step
                # state = torch.tensor([states[i]], dtype=torch.float).to(device)
                # 增加一个维度
                # state = torch.unsqueeze(state, dim=0)
                state_np = np.asarray(states[i], dtype=np.float32)
                state = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).to(device)
                action = agent.take_action(state)[0]
                print("=" * 100)
                # print("state[:11]:{}".format(state[0][0][:11]))
                # print("state[11:]:{}".format(state[0][0][11:]))
                print("action:{}".format(action))
                # 根据选取的动作改变状态，获取收益
                next_state, reward, uav_done, _, info = env.step((action[0], i))
                # 求总收益
                total_reward += reward
                # if agent.action_flag:
                #     print("通过网络计算")
                # else:
                #     print("通过贪心策略计算")
                # print(env.uavs[i].x, env.uavs[i].y, env.uavs[i].z)
                # print(reward)
                if render_mode is not None:
                    env.render()
                    plt.pause(0.01)
                else: pass
                if info['info'] == 1:
                    n_success += 1
                if uav_done:
                    env.unwrapped.uavs[i].done = True
                    n_done += 1
                else:
                    states[i] = next_state
            # 如果一个批次的无人机全都训练完毕
            if n_done >= uavs_num:
                break
            # env.ax.scatter(env.target.x, env.target.y, env.target.z, c='red')
    if render_mode != None:
        plt.pause(600)
    print("=" * 20)
    print(f"成功的次数:{n_success}")
