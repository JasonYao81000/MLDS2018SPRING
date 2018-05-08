# python3 train.py  

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import random
import pickle

# from data_preprocessing import pad_sequences
from seq2seq_model import Seq2Seq_Model
# from bleu_eval import BLEU

# epoch 2313. 

if __name__ == "__main__":

    np.random.seed(9487)
    random.seed(9487)
    tf.set_random_seed(9487)

    tf.app.flags.DEFINE_integer('rnn_size', 256, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('embedding_size', 256, 'Embedding dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer')
    tf.app.flags.DEFINE_float('learning_rate', 0.5, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Max. global gradient norm to clip')

    tf.app.flags.DEFINE_boolean('use_attention', False, 'Enable attention')
    
    tf.app.flags.DEFINE_boolean('use_scheduled', False, 'Enable beam search')
    tf.app.flags.DEFINE_float('sampling_probability', 0.5, 'Size of beam search')
    
    tf.app.flags.DEFINE_boolean('beam_search', False, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 5, 'Size of beam search')

    tf.app.flags.DEFINE_integer('max_encoder_steps', 15, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 500000, 'Sampled data size of training epochs')
    # tf.app.flags.DEFINE_float('validation_split', 0, 'Sampled data size of training epochs')

    tf.app.flags.DEFINE_integer('num_epochs', 150, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS
    
    # num_top_BLEU = 10
    # top_BLEU = []

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    dictionary = pickle.load(open('dictionary.obj', 'rb'))
    conversation_pair = pickle.load(open('conversation_pair.obj', 'rb'))

    index2word_series = pd.Series(index2word)

    print ('There are %d conversation pairs.' %(len(conversation_pair)))
    print ('Spliting data set to training and testing set...')
    # train_conversation_pair = conversation_pair[:int(len(conversation_pair) * (1 - FLAGS.validation_split))]
    # valid_conversation_pair = conversation_pair[int(len(conversation_pair) * (1 - FLAGS.validation_split)):]
    train_conversation_pair = conversation_pair
    valid_conversation_pair = conversation_pair
    print ('There are %d conversation pairs for training.' %(len(train_conversation_pair)))
    print ('There are %d conversation pairs for validation.' %(len(valid_conversation_pair)))


    with tf.Session() as sess:
        model = Seq2Seq_Model(
            rnn_size=FLAGS.rnn_size, 
            num_layers=FLAGS.num_layers, 
            embedding_size=FLAGS.embedding_size, 
            optimizer=FLAGS.optimizer, 
            learning_rate=FLAGS.learning_rate, 
            word_to_idx=word2index, 
            mode='train', 
            max_gradient_norm=FLAGS.max_gradient_norm, 
            use_attention=FLAGS.use_attention, 
            use_scheduled=FLAGS.use_scheduled, 
            sampling_probability=FLAGS.sampling_probability, 
            beam_search=FLAGS.beam_search, 
            beam_size=FLAGS.beam_size,
            max_encoder_steps=FLAGS.max_encoder_steps, 
            max_decoder_steps=FLAGS.max_decoder_steps
        )
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters[%s]..' %(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # for epoch in range(10, FLAGS.num_epochs):
        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()

            # Random sample training set.
            sampled_conversation_pair = random.sample(train_conversation_pair, FLAGS.sample_size)
            # sampled_conversation_pair = train_conversation_pair

            # Random shuffle training set.
            random.shuffle(sampled_conversation_pair)

            for batch_start, batch_end in zip(range(0, len(sampled_conversation_pair), FLAGS.batch_size), range(FLAGS.batch_size, len(sampled_conversation_pair), FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, len(sampled_conversation_pair)), end='\r')

                batch_sampled_conversation_pair = sampled_conversation_pair[batch_start : batch_end]
                batch_source_inputs = [elements[0] for elements in batch_sampled_conversation_pair]
                batch_target_outputs = [FLAGS.max_decoder_steps] * FLAGS.batch_size
                # batch_video_feat_mask = np.zeros((batch_size, max_encoder_steps))
                # batch_target_inputs = np.array(["<BOS> "+ elements[1] for elements in batch_sampled_conversation_pair])
                batch_target_inputs = np.array([elements[1] for elements in batch_sampled_conversation_pair])

                # for index, target_inputs in enumerate(batch_target_inputs):
                #     target_inputs_words = target_inputs.split(" ")
                #     if len(target_inputs_words) < FLAGS.max_decoder_steps:
                #         batch_target_inputs[index] = batch_target_inputs[index] + " <EOS>"
                #     else:
                #         new_target_inputs = ""
                #         for i in range(FLAGS.max_decoder_steps - 1):
                #             new_target_inputs = new_target_inputs + target_inputs_words[i] + " "
                #         batch_target_inputs[index] = new_target_inputs + "<EOS>"

                # for index, source_inputs in enumerate(batch_source_inputs):
                #     source_inputs_words = source_inputs.split(" ")
                #     if len(source_inputs_words) < FLAGS.max_decoder_steps:
                #         batch_source_inputs[index] = batch_source_inputs[index] + " <EOS>"
                #     else:
                #         new_source_inputs = ""
                #         for i in range(FLAGS.max_decoder_steps - 1):
                #             new_source_inputs = new_source_inputs + source_inputs_words[i] + " "
                #         batch_source_inputs[index] = new_source_inputs + "<EOS>"

                batch_source_inputs_index = []
                for source_inputs in batch_source_inputs:
                    words_index = []
                    for source_inputs_words in source_inputs.split(' '):
                        if source_inputs_words in word2index:
                            words_index.append(word2index[source_inputs_words])
                        else:
                            words_index.append(word2index['<UNK>'])
                    batch_source_inputs_index.append(words_index)
                    
                batch_target_inputs_index = []
                for target_inputs in batch_target_inputs:
                    words_index = []
                    for target_inputs_words in target_inputs.split(' '):
                        if target_inputs_words in word2index:
                            words_index.append(word2index[target_inputs_words])
                        else:
                            words_index.append(word2index['<UNK>'])
                    batch_target_inputs_index.append(words_index)

                # batch_source_inputs_matrix = pad_sequences(batch_source_inputs_index, padding='pre', maxlen=FLAGS.max_encoder_steps)
                # batch_target_inputs_matrix = pad_sequences(batch_target_inputs_index, padding='post', maxlen=FLAGS.max_decoder_steps)
                batch_source_inputs_matrix = batch_source_inputs_index
                batch_target_inputs_matrix = batch_target_inputs_index
                # batch_target_inputs_matrix = np.hstack([batch_target_inputs_matrix, np.zeros([len(batch_target_inputs_matrix), 1])]).astype(int)
                batch_source_inputs_length = [len(x) for x in batch_source_inputs_matrix]
                batch_target_inputs_length = [len(x) for x in batch_target_inputs_matrix]

                loss, summary = model.train(
                    sess, 
                    batch_source_inputs_matrix, 
                    batch_source_inputs_length, 
                    batch_target_inputs_matrix, 
                    batch_target_inputs_length)
            
            print()
               
            
            # Random sample training set.
            sampled_conversation_pair = random.sample(valid_conversation_pair, 5)
               
            # Valid on validation set. 
            target_outputs = []
            for batch_start, batch_end in zip(range(0, len(sampled_conversation_pair) + FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, len(sampled_conversation_pair) + FLAGS.batch_size, FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, len(sampled_conversation_pair)), end='\r')
                
                batch_sampled_conversation_pair = sampled_conversation_pair[batch_start : batch_end]
                
                if batch_end < len(sampled_conversation_pair):
                    batch_source_inputs = [elements[0] for elements in batch_sampled_conversation_pair]
                    batch_target_inputs = np.array([elements[1] for elements in batch_sampled_conversation_pair])
                else:
                    batch_source_inputs = [elements[0] for elements in batch_sampled_conversation_pair]
                    batch_target_inputs = [elements[1] for elements in batch_sampled_conversation_pair]
                    for _ in range(batch_end - len(sampled_conversation_pair)):
                        batch_source_inputs.append(sampled_conversation_pair[-1][0])
                        batch_target_inputs.append(sampled_conversation_pair[-1][1])
                    batch_target_inputs = np.array(batch_target_inputs)
                
                batch_target_outputs = [FLAGS.max_decoder_steps] * FLAGS.batch_size

                # for index, target_inputs in enumerate(batch_target_inputs):
                #     target_inputs_words = target_inputs.split(" ")
                #     if len(target_inputs_words) < FLAGS.max_decoder_steps:
                #         batch_target_inputs[index] = batch_target_inputs[index] + " <EOS>"
                #     else:
                #         new_target_inputs = ""
                #         for i in range(FLAGS.max_decoder_steps - 1):
                #             new_target_inputs = new_target_inputs + target_inputs_words[i] + " "
                #         batch_target_inputs[index] = new_target_inputs + "<EOS>"

                # for index, source_inputs in enumerate(batch_source_inputs):
                #     source_inputs_words = source_inputs.split(" ")
                #     if len(source_inputs_words) < FLAGS.max_decoder_steps:
                #         batch_source_inputs[index] = batch_source_inputs[index] + " <EOS>"
                #     else:
                #         new_source_inputs = ""
                #         for i in range(FLAGS.max_decoder_steps - 1):
                #             new_source_inputs = new_source_inputs + source_inputs_words[i] + " "
                #         batch_source_inputs[index] = new_source_inputs + "<EOS>"

                batch_source_inputs_index = []
                for source_inputs in batch_source_inputs:
                    words_index = []
                    for source_inputs_words in source_inputs.split(' '):
                        if source_inputs_words in word2index:
                            words_index.append(word2index[source_inputs_words])
                        else:
                            words_index.append(word2index['<UNK>'])
                    batch_source_inputs_index.append(words_index)
                    
                batch_target_inputs_index = []
                for target_inputs in batch_target_inputs:
                    words_index = []
                    for target_inputs_words in target_inputs.split(' '):
                        if target_inputs_words in word2index:
                            words_index.append(word2index[target_inputs_words])
                        else:
                            words_index.append(word2index['<UNK>'])
                    batch_target_inputs_index.append(words_index)

                # batch_source_inputs_matrix = pad_sequences(batch_source_inputs_index, padding='pre', maxlen=FLAGS.max_encoder_steps)
                # batch_target_inputs_matrix = pad_sequences(batch_target_inputs_index, padding='post', maxlen=FLAGS.max_decoder_steps)
                batch_source_inputs_matrix = batch_source_inputs_index
                batch_target_inputs_matrix = batch_target_inputs_index
                batch_source_inputs_length = [len(x) for x in batch_source_inputs_matrix]
                batch_target_inputs_length = [len(x) for x in batch_target_inputs_matrix]

                batch_target_outputs_index, logits = model.infer(
                    sess, 
                    batch_source_inputs_matrix, 
                    batch_source_inputs_length) 

                if batch_end < len(sampled_conversation_pair):
                    batch_target_outputs_index = batch_target_outputs_index
                else:
                    batch_target_outputs_index = batch_target_outputs_index[:len(sampled_conversation_pair) - batch_start]

                for index, target_output_index in enumerate(batch_target_outputs_index):

                    if FLAGS.beam_search:
                        logits = np.array(logits).reshape(-1, FLAGS.beam_size)
                        max_logits_index = np.argmax(np.sum(logits, axis=0))
                        predict_list = np.ndarray.tolist(target_output_index[0, :, max_logits_index])
                        predict_seq = [index2word[idx] for idx in predict_list]
                        target_output_words = predict_seq
                    else:
                        target_output_index = np.array(target_output_index).reshape(-1)
                        target_output_words = index2word_series[target_output_index]
                        target_output = ' '.join(target_output_words) 

                    target_output = ' '.join(target_output_words)
                    target_output = target_output.replace('<BOS> ', '')
                    target_output = target_output.replace('<EOS>', '')
                    target_output = target_output.replace(' <EOS>', '')
                    target_output = target_output.replace('<PAD> ', '')
                    target_output = target_output.replace(' <PAD>', '')
                    target_output = target_output.replace(' <UNK>', '')
                    target_output = target_output.replace('<UNK> ', '')

                    if (target_output == ""):
                        target_output = '我'
                    # print (batch_source_inputs[index], batch_target_inputs[index], target_output)
                    print ("==================================================")
                    print ("Soruce Input: ", batch_source_inputs[index])
                    print ("Target Input: ", batch_target_inputs[index])
                    print ("Target Output: ", target_output)
                    print ("==================================================")
                    target_outputs.append(target_output)
                        
            # df = pd.DataFrame(np.array([test_video_IDs, test_captions]).T)
            # df.to_csv(output_testset_filename, index=False, header=False)

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
            # # print("Average bleu score is " + str(average))

            # if (len(top_BLEU) < num_top_BLEU):
            #     top_BLEU.append(average)
            #     print ("Saving model with BLEU@1: %.4f ..." %(average))
            #     model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            # else:
            #     if (average > min(top_BLEU)):
            #         # Remove min. BLEU score.
            #         top_BLEU.remove(min(top_BLEU))
            #         top_BLEU.append(average)
            #         print ("Saving model with BLEU@1: %.4f ..." %(average))
            #         model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            # top_BLEU.sort(reverse=True)
            # print ("Top [%d] BLEU: " %(num_top_BLEU), ["%.4f" % x for x in top_BLEU])

            model.saver.save(sess, './models/model', global_step=epoch)
            # print ("Epoch: ", epoch, ", loss: ", loss_val, ', Avg. BLEU@1: ', average, ', Elapsed time: ', str((time.time() - start_time)))
            # print ("Epoch %d/%d, loss: %.6f, Avg. BLEU@1: %.6f, Elapsed time: %.2fs" %(epoch, FLAGS.num_epochs, loss, average, (time.time() - start_time)))
            print ("Epoch %d/%d, loss: %.6f, Elapsed time: %.2fs" %(epoch, FLAGS.num_epochs, loss, (time.time() - start_time)))