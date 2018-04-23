# python3 infer.py ./MLDS_hw2_1_data/testing_data ./testset_output.txt

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import json
import random
import pickle

from seq2seq_model import Seq2Seq_Model
from bleu_eval import BLEU

def inference():
    np.random.seed(9487)
    random.seed(9487)
    tf.set_random_seed(9487)

    test_folder = sys.argv[1]
    output_testset_filename = sys.argv[2]

    test_id_filename = test_folder + '/id.txt'
    test_video_feat_folder = test_folder + '/feat/'

    tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('dim_video_feat', 4096, 'Feature dimensions of each video frame')
    tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Max. global gradient norm to clip')
    
    tf.app.flags.DEFINE_boolean('use_attention', True, 'Enable attention')

    tf.app.flags.DEFINE_boolean('beam_search', True, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 1, 'Size of beam search')
    
    tf.app.flags.DEFINE_integer('max_encoder_steps', 64, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 1450, 'Sampled data size of training epochs')
    tf.app.flags.DEFINE_integer('dim_video_frame', 80, 'Number of frame in each video')

    tf.app.flags.DEFINE_integer('num_epochs', 64, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    index2word_series = pd.Series(index2word)

    print ('Reading testing files...')
    test_video_IDs = []
    with open(test_id_filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            test_video_IDs.append(line)
    
    # # Remove '.avi' from filename.
    # test_video_IDs = [filename[:-4] for filename in test_video_IDs]

    test_video_feat_filenames = os.listdir(test_video_feat_folder)
    test_video_feat_filepaths = [(test_video_feat_folder + filename) for filename in test_video_feat_filenames]

    test_video_feat_dict = {}
    for filepath in test_video_feat_filepaths:
        test_video_feat = np.load(filepath)
        
        sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(test_video_feat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat

    # test_video_caption = json.load(open(testing_label_json_file, 'r'))

    with tf.Session() as sess:
        model = Seq2Seq_Model(
            rnn_size=FLAGS.rnn_size, 
            num_layers=FLAGS.num_layers, 
            dim_video_feat=FLAGS.dim_video_feat, 
            embedding_size=FLAGS.embedding_size, 
            learning_rate=FLAGS.learning_rate, 
            word_to_idx=word2index, 
            mode='decode', 
            max_gradient_norm=FLAGS.max_gradient_norm, 
            use_attention=FLAGS.use_attention, 
            beam_search=FLAGS.beam_search, 
            beam_size=FLAGS.beam_size,
            max_encoder_steps=FLAGS.max_encoder_steps, 
            max_decoder_steps=FLAGS.max_decoder_steps
        )
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

        test_captions = []
        for ID in test_video_IDs:
            test_video_feat = test_video_feat_dict[ID].reshape(1, FLAGS.max_encoder_steps, FLAGS.dim_video_feat)
            test_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size

            test_caption_words_index, logits = model.infer(
                sess, 
                test_video_feat, 
                test_video_frame)

            if FLAGS.beam_search:
                logits = np.array(logits).reshape(-1, FLAGS.beam_size)
                max_logits_index = np.argmax(np.sum(logits, axis=0))
                predict_list = np.ndarray.tolist(test_caption_words_index[0, :, max_logits_index])
                predict_seq = [index2word[idx] for idx in predict_list]
                test_caption_words = predict_seq
            else:
                test_caption_words_index = np.array(test_caption_words_index).reshape(-1)
                test_caption_words = index2word_series[test_caption_words_index]
                test_caption = ' '.join(test_caption_words) 

            test_caption = ' '.join(test_caption_words)
            test_caption = test_caption.replace('<bos> ', '')
            test_caption = test_caption.replace('<eos>', '')
            test_caption = test_caption.replace(' <eos>', '')
            test_caption = test_caption.replace('<pad> ', '')
            test_caption = test_caption.replace(' <pad>', '')
            test_caption = test_caption.replace(' <unk>', '')
            test_caption = test_caption.replace('<unk> ', '')

            if (test_caption == ""):
                test_caption = '.'

            # if ID in ["klteYv1Uv9A_27_33.avi","UbmZAe5u5FI_132_141.avi","wkgGxsuNVSg_34_41.avi", "JntMAcTlOF0_50_70.avi","tJHUH9tpqPg_113_118.avi"] and np.mod(epoch, 5) == 0:
            if ID in ["klteYv1Uv9A_27_33.avi", "UbmZAe5u5FI_132_141.avi", "wkgGxsuNVSg_34_41.avi", "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]:
                print(ID, test_caption)

            test_captions.append(test_caption)
        
        df = pd.DataFrame(np.array([test_video_IDs, test_captions]).T)
        df.to_csv(output_testset_filename, index=False, header=False)
        
        # result = {}
        # with open(output_testset_filename, 'r') as f:
        #     for line in f:
        #         line = line.rstrip()
        #         test_id, caption = line.split(',')
        #         result[test_id] = caption
                
        # bleu=[]
        # for item in test_video_caption:
        #     score_per_video = []
        #     captions = [x.rstrip('.') for x in item['caption']]
        #     score_per_video.append(BLEU(result[item['id']],captions,True))
        #     bleu.append(score_per_video[0])
        # average = sum(bleu) / len(bleu)
        # print("Average bleu score is " + str(average))

if __name__ == "__main__":
    inference()