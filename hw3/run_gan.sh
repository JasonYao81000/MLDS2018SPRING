#!/bin/bash
python3.6 ./hw3_1/main.py --dataset Anime --gan_type WGAN_GP --epoch 10 --batch_size 64 --z_dim 62 --mode train
python3.6 ./hw3_1/main.py --dataset Anime --gan_type WGAN_GP --epoch 9487 --batch_size 64 --z_dim 62 --mode infer