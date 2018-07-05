import sys
import numpy as np
import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')

if __name__ == "__main__":

    # double_dual_rewards = np.load('./hw4_tf_DQN_double_dual/history_recent_avg_reward.npy')
    # double_rewards = np.load('./hw4_tf_DQN_double/history_recent_avg_reward.npy')
    # dual_rewards = np.load('./hw4_tf_DQN_dual/history_recent_avg_reward.npy')
    # none_dqn_rewards = np.load('./hw4_tf_DQN_none/history_recent_avg_reward.npy')
    recent_episode_num = 30

    double_dual_rewards = []
    for index, line in enumerate(open('./hw4_tf_DQN_double_dual/log.txt', 'r')):
        if index == 0 or index == 1:
            continue
        episode, step, reward, recent_reward = line.split(',')
        double_dual_rewards.append(reward)

    double_dual_rewards = np.array(double_dual_rewards)
    print ('double_dual_rewards.shape[0]: {0}'.format(double_dual_rewards.shape[0]))

    double_dual_recent_avg_reward = []
    for i in range(double_dual_rewards.shape[0]):
        recent_rewards = double_dual_rewards[max(0, i - recent_episode_num):i + 1]
        # Convert string list to float list.
        recent_rewards = list(map(float, recent_rewards))
        double_dual_recent_avg_reward.append(sum(recent_rewards) / len(recent_rewards))

    double_rewards = []
    for index, line in enumerate(open('./hw4_tf_DQN_double/log.txt', 'r')):
        if index == 0 or index == 1:
            continue
        episode, step, reward, recent_reward = line.split(',')
        double_rewards.append(reward)

    double_rewards = np.array(double_rewards)
    print ('double_rewards.shape[0]: {0}'.format(double_rewards.shape[0]))

    double_recent_avg_reward = []
    for i in range(double_rewards.shape[0]):
        recent_rewards = double_rewards[max(0, i - recent_episode_num):i + 1]
        # Convert string list to float list.
        recent_rewards = list(map(float, recent_rewards))
        double_recent_avg_reward.append(sum(recent_rewards) / len(recent_rewards))

    dual_rewards = []
    for index, line in enumerate(open('./hw4_tf_DQN_dual/log.txt', 'r')):
        if index == 0 or index == 1:
            continue
        episode, step, reward, recent_reward = line.split(',')
        dual_rewards.append(reward)

    dual_rewards = np.array(dual_rewards)
    print ('dual_rewards.shape[0]: {0}'.format(dual_rewards.shape[0]))

    dual_recent_avg_reward = []
    for i in range(dual_rewards.shape[0]):
        recent_rewards = dual_rewards[max(0, i - recent_episode_num):i + 1]
        # Convert string list to float list.
        recent_rewards = list(map(float, recent_rewards))
        dual_recent_avg_reward.append(sum(recent_rewards) / len(recent_rewards))
        
    none_rewards = []
    for index, line in enumerate(open('./hw4_tf_DQN_none/log.txt', 'r')):
        if index == 0 or index == 1:
            continue
        episode, step, reward, recent_reward = line.split(',')
        none_rewards.append(reward)

    none_rewards = np.array(none_rewards)
    print ('none_rewards.shape[0]: {0}'.format(none_rewards.shape[0]))

    none_recent_avg_reward = []
    for i in range(none_rewards.shape[0]):
        recent_rewards = none_rewards[max(0, i - recent_episode_num):i + 1]
        # Convert string list to float list.
        recent_rewards = list(map(float, recent_rewards))
        none_recent_avg_reward.append(sum(recent_rewards) / len(recent_rewards))

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(double_dual_recent_avg_reward, label='double dual')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last ' + str(recent_episode_num) + ' episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_double_dual.png')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(double_recent_avg_reward, label='double')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last ' + str(recent_episode_num) + ' episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_double.png')
    
    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(dual_recent_avg_reward, label='dual')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last ' + str(recent_episode_num) + ' episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_dual.png')
    
    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(none_recent_avg_reward, label='none')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last ' + str(recent_episode_num) + ' episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_none.png')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(double_dual_recent_avg_reward, label='double dual')
    plt.plot(double_recent_avg_reward, label='double')
    plt.plot(dual_recent_avg_reward, label='dual')
    plt.plot(none_recent_avg_reward, label='none')
    plt.legend(loc='upper left')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last ' + str(recent_episode_num) + ' episodes')
    plt.xlabel('# of time steps')
    plt.savefig('reward_episode_4lines.png')