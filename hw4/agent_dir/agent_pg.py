from agent_dir.agent import Agent
import scipy
import numpy as np
seed = 9487
np.random.seed(seed)

import os
import copy
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def prepro(I):
    # """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    # I = I[35:195]  # crop
    # I = I[::2, ::2, 0]  # downsample by factor of 2
    # I[I == 144] = 0  # erase background (background type 1)
    # I[I == 109] = 0  # erase background (background type 2)
    # I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    # # return I.astype(np.float).ravel()
    # return np.expand_dims(I.astype(np.float), axis=2)
    y = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, (80, 80)) 
    return np.expand_dims(resized.astype(np.float32), axis=2)

class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss'):
            # construct computation graph for loss_clip
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                            - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            # tf.summary.scalar('loss_clip', loss_clip)

            # construct computation graph for loss of entropy bonus
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(states)
            # tf.summary.scalar('entropy', entropy)

            # construct computation graph for loss of value function
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            # tf.summary.scalar('value_difference', loss_vf)

            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy

            # minimize -loss == maximize loss
            loss = -loss
            # tf.summary.scalar('total', loss)

        # self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)
    
    def train(self, states, actions, gaes, rewards, v_preds_next):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.states: states,
                                                               self.Old_Policy.states: states,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, states, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.gradients, feed_dict={self.Policy.states: states,
                                                                       self.Old_Policy.states: states,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})

class Policy_net:
    def __init__(self, name: str, action_size):
        # self.sess = tf.InteractiveSession()
        # self.sampled_actions = tf.placeholder(tf.float32, [None, action_size])
        # self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, [None, 80, 80, 1])

            with tf.variable_scope('policy_net'):
                net = tf.layers.conv2d(
                    inputs=self.states,
                    filters=32,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding='same',
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())

                net = tf.layers.conv2d(
                    inputs=net,
                    filters=64,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding='same',
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())
                
                net = tf.layers.Flatten()(net)
                
                self.v_preds = tf.layers.dense(
                    inputs=net,
                    units=1,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())
                
                net = tf.layers.dense(
                    inputs=net,
                    units=128,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())

                self.act_probs = tf.layers.dense(
                    inputs=net, 
                    units=action_size, 
                    activation=tf.nn.softmax,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())

            self.scope = tf.get_variable_scope().name
            
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

        # self.loss = tf.keras.losses.categorical_crossentropy(self.sampled_actions, self.logits)
        # self.loss = tf.reduce_mean(self.loss)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_op = optimizer.minimize(self.loss)
        # tf.global_variables_initializer().run()

        # self.saver = tf.train.Saver()
        # self.checkpoint_file = os.path.join(checkpoints_dir, 'policy_network.ckpt')

    
    
    def forward_pass(self, states):
        logits = self.sess.run(
            self.logits,
            feed_dict={self.states: states})
        return logits
    
    def train(self, X, Y):
        loss, _ = self.sess.run(
            (self.loss, self.train_op), 
            feed_dict={
                self.states: X,
                self.sampled_actions: Y
            }
        )
        return loss
    
    def act(self, states, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.states: states})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.states: states})

    def get_action_prob(self, states):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.states: states})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        
        super(Agent_PG,self).__init__(env)

        self.sess = tf.InteractiveSession()

        # Load settings from args.
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.discount_factor = args.discount_factor
        self.render = args.render
        self.save_history_period = args.save_history_period
        self.checkpoints_dir = './checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoints_dir, 'policy_network.ckpt')

        # Testing
        self.action_size = 2
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.actions = []
        self.v_preds = []
        self.history_recent_avg_reward = []

        self.env = env
        self.env.seed(seed)

        # self.network = Network(
        #     self.action_size,
        #     self.learning_rate,
        #     checkpoints_dir='./checkpoints')

        self.Policy = Policy_net('policy', self.action_size)
        self.Old_Policy = Policy_net('old_policy', self.action_size)

        self.PPO = PPOTrain(self.Policy, self.Old_Policy, gamma=self.discount_factor)

        self.saver = tf.train.Saver()

        if args.load_checkpoint:
            self.network.load_checkpoint()
            self.history_recent_avg_reward = np.load('./history_recent_avg_reward.npy').tolist()

        if args.test_pg:
            print('Loading trained model...')
            self.load_checkpoint()

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.last_state = None
    
    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        # y = action
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def discount_rewards(self, rewards, discount_factor):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        num_episode = 1
        # Record rewards over 30 episodes.
        recent_episode_num = 30
        recent_rewards = []
        recent_avg_reward = None
        # Record the best Avg. reward.
        best_avg_reward = -87.0

        self.sess.run(tf.global_variables_initializer())

        while True:
            episode_done = False
            sum_reward_episode = 0

            last_state = self.env.reset()
            last_state = prepro(last_state)
            action = self.env.action_space.sample()
            state, _, _, _ = self.env.step(action)
            state = prepro(state)

            num_rounds = 1
            num_actions = 1
            num_win = 0
            num_lose = 0

            while not episode_done:
                if self.render:
                    self.env.render()
                    
                delta_state = state - last_state
                last_state = state

                # prob = self.network.forward_pass(np.expand_dims(delta_state, axis=0))[0]
                # self.probs.append(prob)
                
                # action = np.random.choice(self.action_size, 1, p=prob / sum(prob))[0]
                act, v_pred = self.Policy.act(states=np.expand_dims(delta_state, axis=0), stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                state, reward, episode_done, info = self.env.step(act + 2)

                self.states.append(delta_state)
                self.actions.append(act)
                self.v_preds.append(v_pred)
                self.rewards.append(reward)
                    
                state = prepro(state)
                sum_reward_episode += reward
                num_actions += 1

                # self.remember(delta_state, action, prob, reward)

                if reward != 0:
                    if reward == -1:
                        num_lose += 1
                    elif reward == +1:
                        num_win += 1
                    print ('Round [{:2d}] {:2d}:{:2d}'.format(num_rounds, num_lose, num_win), end='\r')
                    num_rounds += 1

            v_preds_next = self.v_preds[1:] + [0]  # next state of terminate state has 0 state value

            # Get Avg. of recent rewards.
            recent_rewards.append(sum_reward_episode)
            if len(recent_rewards) > recent_episode_num:
                recent_rewards.pop(0)
            recent_avg_reward = sum(recent_rewards) / len(recent_rewards)
            self.history_recent_avg_reward.append(recent_avg_reward)
            print('\rEpisode: {:d}, Actions: {:4d}, reward: {:2.3f}, Avg. reward: {:2.6f}'.format(
                num_episode, num_actions, sum_reward_episode, recent_avg_reward))

            if recent_avg_reward > best_avg_reward:
                print ('[Save Checkpoint] Avg. reward improved from {:2.6f} to {:2.6f}'.format(
                    best_avg_reward, recent_avg_reward))
                best_avg_reward = recent_avg_reward
                self.save_checkpoint()

            if num_episode % self.batch_size == 0:
                # gradients = np.vstack(self.gradients)
                # rewards = np.vstack(self.rewards)
                # rewards = self.discount_rewards(rewards, self.discount_factor)
                # if np.std(rewards) != 0:
                #     rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                # else:
                #     rewards = rewards - np.mean(rewards)
                # gradients *= rewards

                # X = np.vstack([self.states])
                # Y = self.probs + self.learning_rate * np.vstack([gradients])
                # loss = self.network.train(X, Y)
                # self.states, self.probs, self.gradients, self.rewards = [], [], [], []
                gaes = self.PPO.get_gaes(rewards=self.rewards, v_preds=self.v_preds, v_preds_next=v_preds_next)

                states = np.array(self.states).astype(dtype=np.float32)
                actions = np.array(self.actions).astype(dtype=np.int32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()
                rewards = np.array(self.rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

                self.PPO.assign_policy_parameters()

                self.PPO.train(states=states,
                          actions=actions,
                          gaes=gaes,
                          rewards=rewards,
                          v_preds_next=v_preds_next)

                self.states, self.actions, self.v_preds, self.rewards = [], [], [], []

            if num_episode % self.save_history_period == 0:
                np.save('history_recent_avg_reward.npy', np.array(self.history_recent_avg_reward))
                
            num_episode += 1

    def make_action(self, state, test=True):
        """
        Return predicted action of your agent

        Input:
            state: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        state = prepro(state)
        if self.last_state is None:
            delta_state = state
        else:
            delta_state = state - self.last_state
        self.last_state = state

        act, v_pred = self.Policy.act(states=np.expand_dims(delta_state, axis=0), stochastic=False)
        action = np.asscalar(act)

        # prob = self.network.forward_pass(np.expand_dims(delta_state, axis=0))[0]
        # action = np.argmax(prob)
        # # np.random.seed(seed)
        # # action = np.random.choice(self.action_size, 1, p=prob)[0]

        return action + 2