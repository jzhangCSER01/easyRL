"""
定义算法
强化学习算法的模式都比较固定，一般包括 sample（即训练时采样动作），predict（测试时预测动作），update（算法更新）以及保存模型和加载模型等几个方法，
其中对于每种算法 sample 和 update 的方式是不相同，而其他方法就大同小异。
"""

import numpy as np
import torch
import math
from collections import defaultdict


class QLearning(object):

    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr    # 学习率
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda : np.zeros(n_actions))    # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射，即Q表

    def sample_action(self, state):
        """
        采样动作，训练时用
        :param state: 观测到的当前状态
        :return:
        """
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)   # epsilon 是会递减的，这里选择指数递减
        # e-greedy 策略
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])    # 选择 Q(s, a) 最大对应的动作
        else:
            action = np.random.choice(self.n_actions)   # 随机选择动作
        return action

    def predict_action(self, state):
        """
        预测或选择动作，测试时使用
        :param state:
        :return:
        """
        action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, terminated):
        """
        更新 Q-table
        :param state: 当前状态
        :param action: 根据当前状态做出的动作
        :param reward: 得到环境的奖励
        :param next_state: 下一个状态
        :param terminated: 是否终止
        :return:
        """
        Q_predict = self.Q_table[str(state)][action]
        if terminated:  # 终止条件
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)