# python3 train.py ./MLDS_hw2_1_data/testing_data/feat/ ./MLDS_hw2_1_data/testing_label.json ./output_testset.txt

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import random
import pickle

from data_preprocessing import pad_sequences
from seq2seq_model import Seq2Seq_Model
from bleu_eval import BLEU

if __name__ == "__main__":

    np.random.seed(9487)
    random.seed(9487)
    tf.set_random_seed(9487)

    test_video_feat_folder = sys.argv[1]
    testing_label_json_file = sys.argv[2]
    output_testset_filename = sys.argv[3]

    tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('dim_video_feat', 4096, 'Feature dimensions of each video frame')
    tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 29, 'Batch size')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Max. global gradient norm to clip')

    tf.app.flags.DEFINE_boolean('use_attention', True, 'Enable attention')
    
    tf.app.flags.DEFINE_boolean('beam_search', False, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 5, 'Size of beam search')

    tf.app.flags.DEFINE_integer('max_encoder_steps', 64, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 1450, 'Sampled data size of training epochs')
    tf.app.flags.DEFINE_integer('dim_video_frame', 80, 'Number of frame in each video')

    tf.app.flags.DEFINE_integer('num_epochs', 203, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS
    
    num_top_BLEU = 10
    top_BLEU = []

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    # embedding_bias_init_vector = pickle.load(open('embedding_bias_init_vector.obj', 'rb'))
    # ID_caption = pickle.load(open('ID_caption.obj', 'rb'))
    video_IDs = pickle.load(open('video_IDs.obj', 'rb'))
    video_caption_dict = pickle.load(open('video_caption_dict.obj', 'rb'))
    video_feat_dict = pickle.load(open('video_feat_dict.obj', 'rb'))
    index2word_series = pd.Series(index2word)

    print ('Reading testing files...')
    test_video_feat_filenames = os.listdir(test_video_feat_folder)
    test_video_feat_filepaths = [(test_video_feat_folder + filename) for filename in test_video_feat_filenames]
    
    # Remove '.avi' from filename.
    test_video_IDs = [filename[:-4] for filename in test_video_feat_filenames]

    test_video_feat_dict = {}
    for filepath in test_video_feat_filepaths:
        test_video_feat = np.load(filepath)
        
        sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(test_video_feat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat
    
            
    # test_features = [ (file[:-4],np.load(test_video_feat_folder + file)) for file in test_video_feat_filenames]
    
    test_video_caption = json.load(open(testing_label_json_file, 'r'))

    with tf.Session() as sess:
        model = Seq2Seq_Model(
            rnn_size=FLAGS.rnn_size, 
            num_layers=FLAGS.num_layers, 
            dim_video_feat=FLAGS.dim_video_feat, 
            embedding_size=FLAGS.embedding_size, 
            learning_rate=FLAGS.learning_rate, 
            word_to_idx=word2index, 
            mode='train', 
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
            model.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()

            # Random sample ID_caption.
            sampled_ID_caption = []
            for ID in video_IDs:
                sampled_caption = random.sample(video_caption_dict[ID], 1)[0]
                sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
                sampled_video_feat = video_feat_dict[ID][sampled_video_frame]
                sampled_ID_caption.append((sampled_video_feat, sampled_caption))

            # Random shuffle training set.
            random.shuffle(sampled_ID_caption)

            for batch_start, batch_end in zip(range(0, FLAGS.sample_size, FLAGS.batch_size), range(FLAGS.batch_size, FLAGS.sample_size, FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, FLAGS.sample_size), end='\r')

                batch_sampled_ID_caption = sampled_ID_caption[batch_start : batch_end]
                batch_video_feats = [elements[0] for elements in batch_sampled_ID_caption]
                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size
                # batch_video_feat_mask = np.zeros((batch_size, max_encoder_steps))
                batch_captions = np.array(["<bos> "+ elements[1] for elements in batch_sampled_ID_caption])

                for index, caption in enumerate(batch_captions):
                    caption_words = caption.lower().split(" ")
                    if len(caption_words) < FLAGS.max_decoder_steps:
                        batch_captions[index] = batch_captions[index] + " <eos>"
                    else:
                        new_caption = ""
                        for i in range(FLAGS.max_decoder_steps - 1):
                            new_caption = new_caption + caption_words[i] + " "
                        batch_captions[index] = new_caption + "<eos>"

                batch_captions_words_index = []
                for caption in batch_captions:
                    words_index = []
                    for caption_words in caption.lower().split(' '):
                        if caption_words in word2index:
                            words_index.append(word2index[caption_words])
                        else:
                            words_index.append(word2index['<unk>'])
                    batch_captions_words_index.append(words_index)

                batch_captions_matrix = pad_sequences(batch_captions_words_index, padding='post', maxlen=FLAGS.max_decoder_steps)
                # batch_captions_matrix = np.hstack([batch_captions_matrix, np.zeros([len(batch_captions_matrix), 1])]).astype(int)
                batch_captions_length = [len(x) for x in batch_captions_matrix]
               
                loss, summary = model.train(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame, 
                    batch_captions_matrix, 
                    batch_captions_length)
            
            print()
               
            # Validation on testing set. 
            test_captions = []
            # for batch_start, batch_end in zip(range(0, len(test_video_IDs), FLAGS.batch_size), range(FLAGS.batch_size, len(test_video_IDs), FLAGS.batch_size)):
            for batch_start, batch_end in zip(range(0, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, FLAGS.sample_size), end='\r')
                if batch_end < len(test_video_IDs):
                    batch_sampled_ID = np.array(test_video_IDs[batch_start : batch_end])
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]
                else:
                    batch_sampled_ID = test_video_IDs[batch_start : batch_end]
                    for _ in range(batch_end - len(test_video_IDs)):
                        batch_sampled_ID.append(test_video_IDs[-1])
                    batch_sampled_ID = np.array(batch_sampled_ID)
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]

                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size 

                batch_caption_words_index, logits = model.infer(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame) 

                if batch_end < len(test_video_IDs):
                    batch_caption_words_index = batch_caption_words_index
                else:
                    batch_caption_words_index = batch_caption_words_index[:len(test_video_IDs) - batch_start]

                for index, test_caption_words_index in enumerate(batch_caption_words_index):

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
                    if batch_sampled_ID[index] in ["klteYv1Uv9A_27_33.avi", "UbmZAe5u5FI_132_141.avi", "wkgGxsuNVSg_34_41.avi", "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]:
                        print(batch_sampled_ID[index], test_caption)
                    test_captions.append(test_caption)
                        
            df = pd.DataFrame(np.array([test_video_IDs, test_captions]).T)
            df.to_csv(output_testset_filename, index=False, header=False)

            result = {}
            with open(output_testset_filename, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    test_id, caption = line.split(',')
                    result[test_id] = caption
                    
            bleu=[]
            for item in test_video_caption:
                score_per_video = []
                captions = [x.rstrip('.') for x in item['caption']]
                score_per_video.append(BLEU(result[item['id']],captions,True))
                bleu.append(score_per_video[0])
            average = sum(bleu) / len(bleu)
            # print("Average bleu score is " + str(average))

            if (len(top_BLEU) < num_top_BLEU):
                top_BLEU.append(average)
                print ("Saving model with BLEU@1: %.4f ..." %(average))
                model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            else:
                if (average > min(top_BLEU)):
                    # Remove min. BLEU score.
                    top_BLEU.remove(min(top_BLEU))
                    top_BLEU.append(average)
                    print ("Saving model with BLEU@1: %.4f ..." %(average))
                    model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            top_BLEU.sort(reverse=True)
            print ("Top [%d] BLEU: " %(num_top_BLEU), ["%.4f" % x for x in top_BLEU])

            # print ("Epoch: ", epoch, ", loss: ", loss_val, ', Avg. BLEU@1: ', average, ', Elapsed time: ', str((time.time() - start_time)))
            print ("Epoch %d/%d, loss: %.6f, Avg. BLEU@1: %.6f, Elapsed time: %.2fs" %(epoch, FLAGS.num_epochs, loss, average, (time.time() - start_time)))