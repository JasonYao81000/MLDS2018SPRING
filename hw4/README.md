# MLDS2018SPRING/hw4

# Table of Contents
<!--ts-->
   0. [Requirements](#4-0-requirements)
   1. [Policy Gradient](#4-1-policy-gradient)
      * [Introduction](#introduction)
      * [Baseline](#baseline)
      * [Testing Policy Gradient](#testing-policy-gradient)
      * [Rewards in 30 Episodes](#rewards-in-30-episodes)
      * [Learning Curve](#learning-curve)
   2. [Deep Q Learning](#4-2-deep-q-learning)
      * [Introduction](#introduction-1)
      * [Baseline](#baseline-1)
      * [Testing Policy Gradient](#testing-policy-gradient-1)
      * [Rewards in 100 Episodes](#rewards-in-100-episodes)
      * [Learning Curve](#learning-curve-1)
   3. [Actor-Critic](#4-3-actor-critic)
      * [Introduction](#introduction-2)
      * [Learning Curve of Actor-Critic and A3C on Pong](#learning-curve-of-actor-critic-and-a3c-on-pong)
      * [Learning Curve of Actor-Critic and A3C on Breakout](#learning-curve-of-actor-critic-and-a3c-on-breakout)
<!--te-->

# 4-0. Requirements
```
atari-py==0.1.1
gym==0.10.5
matplotlib==2.2.2
numpy==1.14.3
opencv-python==3.4.0.12
scipy==1.1.0
tensorflow-gpu==1.6.0
```

# 4-1. Policy Gradient

## Introduction
* Game Playing: Pong
* Implement an agent to play Atari games using Deep Reinforcement Learning.
* In this homework, you are required to implement Policy Gradient.
* The Pong environment is used in this homework.
* Improvements to Policy Gradient: 
  - [ ] Variance Reduction
  - [ ] Natural Policy Gradient
  - [ ] Trust Region Policy Optimization
  - [x] **Proximal Policy Optimization**
* Training Hint
  - [x] Reward normalization (More stable)
  - [x] Action space reduction (Only up and down)

## Baseline
* Getting averaging reward in 30 episodes over **3** in **Pong**
* Without OpenAI’s Atari wrapper & reward clipping
* Improvements to Policy Gradient are allowed

## Testing Policy Gradient
`$ python3.6 test.py --test_pg`

## Rewards in 30 Episodes
```
ep 0, reward: 18.000000
ep 1, reward: 17.000000
ep 2, reward: 18.000000
ep 3, reward: 19.000000
ep 4, reward: 13.000000
ep 5, reward: 17.000000
ep 6, reward: 9.000000
ep 7, reward: 20.000000
ep 8, reward: 16.000000
ep 9, reward: 17.000000
ep 10, reward: 12.000000
ep 11, reward: 19.000000
ep 12, reward: 16.000000
ep 13, reward: 18.000000
ep 14, reward: 18.000000
ep 15, reward: 19.000000
ep 16, reward: 15.000000
ep 17, reward: 12.000000
ep 18, reward: 15.000000
ep 19, reward: 21.000000
ep 20, reward: 17.000000
ep 21, reward: 18.000000
ep 22, reward: 13.000000
ep 23, reward: 18.000000
ep 24, reward: 19.000000
ep 25, reward: 20.000000
ep 26, reward: 17.000000
ep 27, reward: 15.000000
ep 28, reward: 14.000000
ep 29, reward: 14.000000
Run 30 episodes
Mean: 16.466666666666665
```

## Learning Curve

* Learning Curve of Original Policy Gradient
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/avg_reward_pg.png" width="100%">

* Learning Curve of Policy Gradient with Proximal Policy Optimization (PPO)
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/avg_reward_pg_ppo.png" width="100%">

* Comparison of Original PG and PG with PPO
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/avg_reward_pg_ppo_2lines.png" width="100%">

# 4-2. Deep Q Learning

## Introduction
* Game Playing: Breakout
* Implement an agent to play Atari games using Deep Reinforcement Learning.
* In this homework, you are required to implement Deep Q-Learning (DQN).
* The Breakout environment is used in this homework.
* Improvements to DQN: 
  - [x] **Double Q-Learning**
  - [x] **Dueling Network**
  - [ ] Prioritized Replay Memory
  - [ ] Noisy DQN
  - [ ] Distributional DQN
* Training Hint
  * The action should act ε-greedily
    - [x] Random action with probability ε
    - [ ] Also in testing
  * Linearly decline ε from 1.0 to some small value, say 0.025
    - [x] Decline per step
    - [x] Randomness is for exploration, agent is weak at start
  * Hyperparameters
    - [x] Replay Memory Size 10000
    - [x] Perform Update Current Network Step 4
    - [x] Perform Update Target Network Step 1000
    - [x] Learning Rate 1e-4
    - [x] Batch Size 32

## Baseline
* Getting averaging reward in 100 episodes over **40** in **Breakout**
* With OpenAI’s Atari wrapper & reward clipping
  * We will unclip the reward when testing

## Testing Deep Q Learning
`$ python3 test.py --test_dqn`

## Rewards in 100 Episodes
```
ep 0, reward: 
ep 1, reward: 
...
ep 99, reward: 
Run 100 episodes
Mean: 
```

## Learning Curve

* Learning Curve of DQN
<img src="" width="100%">

* Learning Curve of Dual DQN
<img src="" width="100%">

* Learning Curve of Double DQN
<img src="" width="100%">

* Learning Curve of Double Dual DQN
<img src="" width="100%">

* Comparison of DQN, Dual DQN, Double DQN and Double Dual DQN
<img src="" width="100%">

# 4-3. Actor-Critic

## Introduction
* Game Playing: Pong and Breakout
* Implement an agent to play Atari games using Actor-Critic.
* Improvements to Actor-Critic: 
  - [ ] DDPG (Deep Deterministic Policy Gradient)
  - [ ] ACER (Sample Efficient Actor-Critic with Experience Replay)
  - [x] **A3C (Asynchronous Advantage Actor-Critic)**
  - [ ] A2C (Synchronous Advantage Actor Critic)
  - [ ] ACKTR (Actor Critic using Kronecker-Factored Trust Region)

## Learning Curve of Actor-Critic and A3C on Pong

* Learning Curve (Reward v.s. Episode) of Actor-Critic
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/reward_episode_ac.png" width="100%">

* Learning Curve (Reward v.s. Episode) of A3C
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/reward_episode_a3c.png" width="100%">

* Comparison (Reward v.s. Episode) of Actor-Critic and A3C
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/reward_episode_ac%26a3c.png" width="100%">

* Learning Curve (Reward v.s. Time) of Actor-Critic
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/reward_time_ac.png" width="100%">

* Learning Curve (Reward v.s. Time) of A3C
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/reward_time_a3c.png" width="100%">

* Comparison (Reward v.s. Time) of Actor-Critic and A3C
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/reward_time_ac%26a3c.png" width="100%">

## Learning Curve of Actor-Critic and A3C on Breakout

* Learning Curve (Reward v.s. Episode) of Actor-Critic
<img src="" width="100%">

* Learning Curve (Reward v.s. Episode) of A3C
<img src="" width="100%">

* Comparison (Reward v.s. Episode) of Actor-Critic and A3C
<img src="" width="100%">

* Learning Curve (Reward v.s. Time) of Actor-Critic
<img src="" width="100%">

* Learning Curve (Reward v.s. Time) of A3C
<img src="" width="100%">

* Comparison (Reward v.s. Time) of Actor-Critic and A3C
<img src="" width="100%">
