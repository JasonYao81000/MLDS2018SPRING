import os

## GAN Variants
# from GAN import GAN
# from CGAN import CGAN
# from infoGAN import infoGAN
# from ACGAN import ACGAN
# from EBGAN import EBGAN
# from WGAN import WGAN
from WGAN_GP import WGAN_GP
# from DRAGAN import DRAGAN
# from LSGAN import LSGAN
# from BEGAN import BEGAN

## VAE Variants
# from VAE import VAE
# from CVAE import CVAE

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

import numpy as np

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA', 'Anime'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='./hw3_2/checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='./hw3_2/results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./hw3_2/logs',
                        help='Directory name to save training logs')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')
    parser.add_argument('--testing_tags', type=str, default='./AnimeDataset/sample_testing_text.txt',
                        help='train or infer')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # Fix random seed.
    tf.set_random_seed(9487)
    
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    # models = [GAN, CGAN, infoGAN, ACGAN, EBGAN, WGAN, WGAN_GP, DRAGAN,
    #           LSGAN, BEGAN, VAE, CVAE]
    models = [WGAN_GP]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir, 
                            mode=args.mode)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()
        if args.mode == 'train':
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

            # # visualize learned generator
            # gan.visualize_results(args.epoch-1)
            # print(" [*] Testing finished!")
        elif args.mode == 'infer':
            tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
                        'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
                        'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                        'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
            testing_tags_txt = open(args.testing_tags, 'r').readlines()

            test_labels = np.zeros((args.batch_size, len(tag_dict)))
            for line in testing_tags_txt:
                id, tags = line.split(',')
                label = np.zeros(len(tag_dict))
                
                for i in range(len(tag_dict)):
                    if tag_dict[i] in tags:
                        label[i] = 1
                test_labels[int(id) - 1] = label
                
            # visualize learned generator
            gan.infer(test_labels)
            print(" [*] Infer finished!")

if __name__ == '__main__':
    main()
