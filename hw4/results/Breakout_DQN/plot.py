import sys
import numpy as np
import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')

if __name__ == "__main__":

    double_dual_rewards = np.load('./hw4_tf_DQN_double_dual/history_recent_avg_reward.npy')
    double_rewards = np.load('./hw4_tf_DQN_double/history_recent_avg_reward.npy')
    dual_rewards = np.load('./hw4_tf_DQN_dual/history_recent_avg_reward.npy')
    none_dqn_rewards = np.load('./hw4_tf_DQN_none/history_recent_avg_reward.npy')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(double_dual_rewards, label='double dual')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 100 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_double_dual.png')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(double_rewards, label='double')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 100 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_double.png')
    
    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(dual_rewards, label='dual')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 100 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_dual.png')
    
    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(none_dqn_rewards, label='none')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 100 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_none.png')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(double_dual_rewards, label='double dual')
    plt.plot(double_rewards, label='double')
    plt.plot(dual_rewards, label='dual')
    plt.plot(none_dqn_rewards, label='none')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 100 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_4lines.png')