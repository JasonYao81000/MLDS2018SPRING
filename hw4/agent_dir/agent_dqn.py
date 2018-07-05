from agent_dir.agent import Agent
import numpy as np
import random
import tensorflow as tf
seed = 9487
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import pickle
from collections import deque
import os

# from keras.models import Sequential
# from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Lambda
# from keras.optimizers import Adam, RMSprop
# from keras.layers import Conv2D
# from keras.backend.tensorflow_backend import set_session
# from keras.models import load_model
# from keras import backend as K

# def loss_function(y, label):
#     error = tf.keras.backend.abs(y - label)
#     quadratic_part = tf.keras.backend.clip(error, 0.0, 1.0)
#     linear_part = error - quadratic_part
#     return tf.keras.backend.mean(0.5*tf.keras.backend.square(quadratic_part)+linear_part, axis=-1)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        self.dqn_double = False
        self.dqn_duel = False
        
        super(Agent_DQN,self).__init__(env)
        self.history_recent_avg_reward = []
        self.save_history_period = 10

        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=10000)
        self.total_episode = 100000
        self.gamma = 0.95   
        self.epsilon = 1.0  
        self.epsilon_min = 0.05
        self.epsilon_step = 100000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_step
        self.learning_rate = 0.0001
        self.checkpoints_dir = './checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoints_dir, 'dqn.ckpt')

        self.sess = tf.InteractiveSession()

        self._build_model()
        self.model_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.target_model_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)
        
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for model_w, target_w in zip(self.model_weights, self.target_model_weights):
                self.assign_ops.append(tf.assign(target_w, model_w))	

        self.target = tf.placeholder(tf.float32, (None, self.action_size), name="target")
        self.loss = tf.reduce_mean((tf.keras.losses.mean_squared_error(self.Q, self.target)))
        self.train_op = tf.keras.optimizers.RMSprop(lr=self.learning_rate).get_updates(self.loss, self.model_weights)

        self.saver = tf.train.Saver() 
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.assign_ops)

        if args.test_dqn:
            #you can load your model here
            print("Loading checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)
        
    def _build_model(self):
        self.state = tf.placeholder(tf.float32, (None, 84, 84, 4), name="state")
            
        with tf.variable_scope('model'):
            net = tf.layers.conv2d(
            inputs=self.state,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid',
            data_format='channels_last',
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            bias_initializer=tf.zeros_initializer())
        
            net = tf.layers.conv2d(
                inputs=net,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='valid',
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=tf.zeros_initializer())

            net = tf.layers.conv2d(
                inputs=net,
                filters=64,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='valid',
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=tf.zeros_initializer())

            net = tf.layers.Flatten()(net)

            net = tf.layers.dense(
                inputs=net,
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=tf.zeros_initializer())
            
            if self.dqn_duel:
                y = tf.layers.dense(
                    inputs=net,
                    units=self.action_size + 1,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())
                self.Q = tf.keras.layers.Lambda(lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.mean(a[:, 1:], keepdims=True), output_shape=(self.action_size,))(y)
            else:
                self.Q = tf.layers.dense(
                        inputs=net,
                        units=self.action_size,
                        activation=None,
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                        bias_initializer=tf.zeros_initializer())
            self.scope = tf.get_variable_scope().name

        with tf.variable_scope('target_model'):
            target_net = tf.layers.conv2d(
            inputs=self.state,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid',
            data_format='channels_last',
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            bias_initializer=tf.zeros_initializer())
        
            target_net = tf.layers.conv2d(
                inputs=target_net,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='valid',
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=tf.zeros_initializer())

            target_net = tf.layers.conv2d(
                inputs=target_net,
                filters=64,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='valid',
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=tf.zeros_initializer())

            target_net = tf.layers.Flatten()(target_net)

            target_net = tf.layers.dense(
                inputs=target_net,
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                bias_initializer=tf.zeros_initializer())
            
            if self.dqn_duel:
                target_y = tf.layers.dense(
                    inputs=target_net,
                    units=self.action_size + 1,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                    bias_initializer=tf.zeros_initializer())
                self.target_Q = tf.keras.layers.Lambda(lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.mean(a[:, 1:], keepdims=True), output_shape=(self.action_size,))(target_y)
            else:
                self.target_Q = tf.layers.dense(
                        inputs=target_net,
                        units=self.action_size,
                        activation=None,
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                        bias_initializer=tf.zeros_initializer())
            self.target_scope = tf.get_variable_scope().name

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, test):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
        if test:
            act_values = self.sess.run(self.Q, {self.state:np.expand_dims(state, axis=0)})
            if self.t > 2500:
                return np.random.choice(self.action_size, 1, p=act_values)[0]
            return np.argmax(act_values[0])
        else:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.sess.run(self.Q, {self.state:np.expand_dims(state, axis=0)})
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        targets = self.sess.run(self.Q, {self.state:states})
        next_actions = self.sess.run(self.Q, {self.state:next_states})
        target_actions = self.sess.run(self.target_Q, {self.state:next_states})
        
        if not self.dqn_double:
            targets[range(batch_size),actions] = rewards + (1 - dones) * self.gamma * np.max(target_actions, axis=1)
        else:
            targets[range(batch_size),actions] = rewards + (1 - dones) * self.gamma * target_actions[range(batch_size), np.argmax(next_actions, axis=1)]

        _, loss = self.sess.run([self.train_op, self.loss], 
            feed_dict={self.state: states, self.target: targets})
        return loss, np.amax(next_actions)
    
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.t = 0

    def train(self):
        """
        Implement your training algorithm here
        """
        log = open('./log.txt','w')
        print('double_dqn:', self.double_dqn, ', dual_dqn:', self.dual_dqn, end='\n', file=log, flush=True)
        log.write('episode,step,reward,recent_reward\n')
        
        total_action = 0
        recent_episode_num = 100
        recent_rewards = []
        recent_avg_reward = 0.0
        best_avg_reward = -87.0
        for episode in range(self.total_episode):
            state = self.env.reset()
            done = False
            num_action = 0
            total_loss = 0
            sum_reward = 0
            total_maxQ = [0]
            
            while done!= True:
                action = self.act(state, False)
                next_state, reward, done, _ = self.env.step(action)
                sum_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                loss = 0
                maxQ = 0
                if total_action > 2000 and j % 4 == 0:
                    loss, maxQ = self.replay(32)
                    total_loss += loss
                    total_maxQ.append(maxQ)
                if j % 1000 == 0 and j > 0:
                    self.sess.run(self.assign_ops)
            
                total_action += 1
                num_action += 1
                
            recent_rewards.append(sum_reward)
            if len(recent_rewards) > recent_episode_num:
                recent_rewards.pop(0)
            recent_avg_reward = sum(recent_rewards) / len(recent_rewards)
            self.history_recent_avg_reward.append(recent_avg_reward)
            print("ep: %5d / step: %5d / reward: %f / Avg. reward: %2.6f / total_action: %5d / loss: %f / Q: %f"%(episode, num_action, sum_reward, recent_avg_reward, total_action, total_loss/num_action, np.max(total_maxQ)))
            print('{:6d},{:5d},{:4f},{:2.6f}'.format(episode, num_action, sum_reward, recent_avg_reward), end='\n', file=log, flush=True)

            if episode % self.save_history_period == 0:
                np.save('history_recent_avg_reward.npy', np.array(self.history_recent_avg_reward))

            if recent_avg_reward > best_avg_reward:
                print ('[Save Checkpoint] Avg. reward improved from {:2.6f} to {:2.6f}'.format(
                    best_avg_reward, recent_avg_reward))
                best_avg_reward = recent_avg_reward
                print("Saving checkpoint...")
                self.saver.save(self.sess, self.checkpoint_file)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        action = self.act(observation, True)
        self.t += 1
        return action

