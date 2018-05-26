#!/bin/bash
# python3.6 ./hw3_2/main.py --dataset Anime --gan_type WGAN_GP --epoch 200 --batch_size 128 --z_dim 128 --mode train
# bash run_cgan.sh ./AnimeDataset/testing_tags.txt
python3.6 ./hw3_2/main.py --dataset Anime --gan_type WGAN_GP --epoch 200 --batch_size 128 --z_dim 128 --mode infer --testing_tags $1