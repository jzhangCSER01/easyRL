"""
Solving FrozenLake environment using Policy-Iteration.
This script requires the latest gym version 0.26.2
"""

import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register
nA = 4  # 动作空间
nS = 4 * 4  # 状态空间


def run_episode(env, policy, gamma = 1.0, render = False):
    """
    Runs an episode and return the total reward
    :param env: 环境
    :param policy: 策略函数
    :param gamma: 折扣因子
    :param render: 是否渲染
    :return:
    """
    observation, info = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        observation, reward, terminated, truncated, info = env.step(int(policy[observation]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if terminated or truncated:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    """
    对策略函数进行打分
    :param env: 环境
    :param policy: 策略函数
    :param gamma: 折扣因子
    :param n:
    :return:
    """
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, gamma = 1.0):
    """
    Extract the policy given a value-function
    :param v:
    :param gamma:
    :return:
    """
    policy = np.zeros(nS)
    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma = 1.0):
    """
    Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in items of v[s]
    and solve them to find the value function.
    :param env: 环境
    :param policy: 策略
    :param gamma: 折扣因子
    :return:
    """
    v = np.zeros(nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if np.sum(np.fabs(prev_v - v)) <= eps:
            # value converged
            break
    return v


def policy_iteration(env, gamma = 1.0):
    """
    Policy-Iteration algorithm
    :param env: 环境
    :param gamma: 折扣因子
    :return:
    """

    policy = np.random.choice(nA, size=(nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if np.all(policy == new_policy):
            print(f'Policy-Iteration converged at step {i + 1}.')
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    optimal_policy = policy_iteration(env, gamma=1.0)
    scores = evaluate_policy(env, optimal_policy, gamma=1.0)
    print(f'Average scores = {np.mean(scores)}')