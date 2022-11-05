import gym
import numpy as np


class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action   # 返回动作

    def learn(self, *args): # 学习
        pass


def play_montecarlo(env, agent, render=False, train=False):
    """
    play_montecarlo() 函数可以让智能体和环境交互一个回合
    :param env: 环境类
    :param agent: 智能体类
    :param render: bool 型变量，指示在运行过程中是否要图形化显示，如果函数参数 render为 True，
    那么在交互过程中会调用 env.render() 以显示图形界面，而这个界面可以通过调用 env.close()关闭。
    :param train: bool 型变量，指示在运行过程中是否训练智能体，在训练过程中应当设置为 True，以调用 agent.learn()函数；
    在测试过程中应当设置为 False，使得智能体不变。
    :return: episode_reward，是 float 型的数值，表示智能体与环境交互一个回合的回合总奖励。
    """
    episode_reward = 0. # 记录回合总奖励，初始化为 0
    observation, info = env.reset()   # 重置游戏环境，开始新回合
    while True: # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()    # 显示图形界面，图形界面可以用 env.close() 语句关闭
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)    # 执行动作
        episode_reward += reward    # 收集回合奖励
        if train:   # 判断是否训练智能体
            agent.learn(observation, action, reward, terminated, truncated)  # 学习
        if terminated or truncated:    # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward   # 返回回合总奖励


env = gym.make('MountainCar-v0', render_mode='human')
agent = BespokeAgent(env)
print(f'观测空间 = {env.observation_space}')
print(f'动作空间 = {env.action_space}')
print(f'观测范围 = {env.observation_space.low}~{env.observation_space.high}')
print(f'动作数 = {env.action_space.n}')
# env.seed(0) # 设置随机数种子，只是为了让结果可以精确复现，一般情况可删去
episode_reward = play_montecarlo(env, agent, render=True)
print(f'回合奖励 = {episode_reward}')
env.close() # 关闭图形界面