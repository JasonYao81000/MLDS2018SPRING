import numpy as np
import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')

if __name__ == "__main__":
    rewards_pg = np.load('./history_recent_avg_reward_pg.npy')
    rewards_pg_ppo = np.load('./history_recent_avg_reward_pg_ppo.npy')

    print ('rewards_pg.shape[0]: {0}'.format(rewards_pg.shape[0]))
    print ('rewards_pg_ppo.shape[0]: {0}'.format(rewards_pg_ppo.shape[0]))
    print ('Clipping length of rewards to 10000...')
    if (rewards_pg.shape[0] > 10000):
        rewards_pg = rewards_pg[:10000]
    if (rewards_pg_ppo.shape[0] > 10000):
        rewards_pg_ppo = rewards_pg_ppo[:10000]
    print ('rewards_pg.shape[0]: {0}'.format(rewards_pg.shape[0]))
    print ('rewards_pg_ppo.shape[0]: {0}'.format(rewards_pg_ppo.shape[0]))

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(rewards_pg)
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 30 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('avg_reward_pg.png')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(rewards_pg_ppo)
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 30 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('avg_reward_pg_ppo.png')

    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(rewards_pg, label='pg')
    plt.plot(rewards_pg_ppo, label='pg_ppo')
    plt.legend(loc='lower right')
    plt.title('Reward v.s. Episode')
    plt.ylabel('Average reward in last 30 episodes')
    plt.xlabel('# of time steps')
    plt.savefig('avg_reward_pg_ppo_2lines.png')