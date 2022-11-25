"""
定义训练 训练及测试 agent
"""


def train(cfg, env, agent):
    print('开始训练!')
    print(f'环境: {cfg.env_name}, 算法: {cfg.algo_name}, 设备: {cfg.device}')
    rewards = []    # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0   # 记录每个回合的奖励
        state, info = env.reset(seed=cfg.seed)  # 重置环境，即开始新的回合
        while True:
            action = agent.sample_action(state) # 根据算法采样一个动作
            next_state, reward, terminated, truncated, info = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, terminated)  # Q 学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        if(i_ep + 1) % 20 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")
    print('完成训练!')
    return {'rewards': rewards}


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0   # 记录每个回合的奖励
        state, info = env.reset(seed=cfg.seed)  # 重置环境，即开始新的回合
        while True:
            action = agent.predict_action(state) # 根据算法采样一个动作
            next_state, reward, terminated, truncated, info = env.step(action)  # 与环境进行一次动作交互
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试!')
    return {'rewards': rewards}