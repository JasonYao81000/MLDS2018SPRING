# MLDS2018SPRING/hw4

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
  * Variance Reduction
  * Natural Policy Gradient
  * Trust Region Policy Optimization
  * **Proximal Policy Optimization** (We used)

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

## Learning Curve of Original Policy Gradient
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/avg_reward_pg.png" width="100%">

## Learning Curve of Policy Gradient with Proximal Policy Optimization (PPO)
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/avg_reward_pg_ppo.png" width="100%">

## Comparison of Original PG and PG with PPO
<img src="https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw4/results/avg_reward_pg_ppo_2lines.png" width="100%">

# 4-2. Deep Q Learning
## Testing Deep Q Learning
`$ python3 test.py --test_dqn`
