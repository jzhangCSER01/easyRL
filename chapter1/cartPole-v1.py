"""
CartPole-v0 环境有两个动作：将小车向左移动和将小车向右移动。
我们还可以得到观测：小车当前的位置，小车当前往左、往右移的速度，杆的角度以及杆的最高点（顶端）的速度。
这里有奖励的定义，如果能多走一步，我们就会得到一个奖励（奖励值为1），所以我们需要存活尽可能多的时间来得到更多的奖励。
当杆的角度大于某一个角度（没能保持平衡），或者小车的中心到达图形界面窗口的边缘，或者累积步数大于200，游戏就结束了，我们就输了。
所以智能体的目的是控制杆，让它尽可能地保持平衡以及尽可能保持在环境的中央。
"""

import gym  # 导入 Gym 的 Python 接口环境
import time
env = gym.make('CartPole-v1', render_mode='rgb_array')   # 构建实验环境
env.reset() # 重置一个回合
for _ in range(1000):
    env.render()    # 显示图形界面
    action = env.action_space.sample()  # 从动作空间中随机选取一个动作
    observation, reward, terminated, truncated, info = env.step(action)    # 用于提交动作，括号内是具体的动作
    time.sleep(0.02)
    print(observation)
env.close() # 关闭环境