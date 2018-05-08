# python3 infer.py ./test_input.txt ./test_output.txt

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import random
import pickle

from seq2seq_model import Seq2Seq_Model
from data_preprocessing import pad_sequences

def inference():
    np.random.seed(9487)
    random.seed(9487)
    tf.set_random_seed(9487)

    input_testset_filename = sys.argv[1]
    output_testset_filename = sys.argv[2]

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
    
    tf.app.flags.DEFINE_boolean('beam_search', True, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 7, 'Size of beam search')

    tf.app.flags.DEFINE_integer('max_encoder_steps', 15, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 500000, 'Sampled data size of training epochs')
    # tf.app.flags.DEFINE_float('validation_split', 0, 'Sampled data size of training epochs')

    tf.app.flags.DEFINE_integer('num_epochs', 150, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    dictionary = pickle.load(open('dictionary.obj', 'rb'))
    # conversation_pair = pickle.load(open('conversation_pair.obj', 'rb'))

    index2word_series = pd.Series(index2word)

    print ('Reading testing files...')
    test_inputs = []
    with open(input_testset_filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            test_inputs.append((line, line))
    
    
    # valid_conversation_pair = conversation_pair[int(len(conversation_pair) * (1 - FLAGS.validation_split)):]

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
            raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

        sampled_conversation_pair = test_inputs

        target_outputs = []
        for batch_start, batch_end in zip(range(0, len(sampled_conversation_pair) + FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, len(sampled_conversation_pair) + FLAGS.batch_size, FLAGS.batch_size)):
            print ("%04d/%04d" %(batch_end, len(sampled_conversation_pair)), end='\r')
            
            batch_sampled_conversation_pair = sampled_conversation_pair[batch_start : batch_end]
            
            if batch_end < len(sampled_conversation_pair):
                batch_source_inputs = [elements[0] for elements in batch_sampled_conversation_pair]
                batch_target_inputs = np.array(["<BOS> "+ elements[1] for elements in batch_sampled_conversation_pair])
            else:
                batch_source_inputs = [elements[0] for elements in batch_sampled_conversation_pair]
                batch_target_inputs = ["<BOS> "+ elements[1] for elements in batch_sampled_conversation_pair]
                for _ in range(batch_end - len(sampled_conversation_pair)):
                    batch_source_inputs.append(sampled_conversation_pair[-1][0])
                    batch_target_inputs.append(sampled_conversation_pair[-1][1])
                batch_target_inputs = np.array(batch_target_inputs)
            
            batch_target_outputs = [FLAGS.max_decoder_steps] * FLAGS.batch_size

            for index, target_inputs in enumerate(batch_target_inputs):
                target_inputs_words = target_inputs.split(" ")
                if len(target_inputs_words) < FLAGS.max_decoder_steps:
                    batch_target_inputs[index] = batch_target_inputs[index] + " <EOS>"
                else:
                    new_target_inputs = ""
                    for i in range(FLAGS.max_decoder_steps - 1):
                        new_target_inputs = new_target_inputs + target_inputs_words[i] + " "
                    batch_target_inputs[index] = new_target_inputs + "<EOS>"

            for index, source_inputs in enumerate(batch_source_inputs):
                source_inputs_words = source_inputs.split(" ")
                if len(source_inputs_words) < FLAGS.max_decoder_steps:
                    batch_source_inputs[index] = batch_source_inputs[index] + " <EOS>"
                else:
                    new_source_inputs = ""
                    for i in range(FLAGS.max_decoder_steps - 1):
                        new_source_inputs = new_source_inputs + source_inputs_words[i] + " "
                    batch_source_inputs[index] = new_source_inputs + "<EOS>"

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

            batch_source_inputs_matrix = pad_sequences(batch_source_inputs_index, padding='pre', maxlen=FLAGS.max_encoder_steps)
            batch_target_inputs_matrix = pad_sequences(batch_target_inputs_index, padding='post', maxlen=FLAGS.max_decoder_steps)
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
                    # Select max logit.
                    # print (np.array(logits).shape)
                    # Reshape (3, batch_size, max_step, beam_size) to (batch_size, max_step, beam_size).
                    single_logits = np.array(logits)[0, :]
                    # print (np.array(logits).shape)
                    # logits = np.array(logits)[0, :].reshape(-1, FLAGS.beam_size)
                    # print (np.array(logits).shape)
                    # Find max. logits in [beam_size] which sumed over [max_step].
                    max_logits_index = np.argmax(np.sum(single_logits, axis=1), axis=1)
                    # print (max_logits_index)
                    # print (max_logits_index[index])
                    # print (target_output_index.shape)
                    # print (target_output_index[:, max_logits_index])
                    # print (target_output_index[:, max_logits_index].shape)
                    predict_list = np.ndarray.tolist(target_output_index[:, max_logits_index[index]])
                    # print (predict_list)
                    
                    # Select max. length.
                    # target_output_index = np.ndarray.tolist(target_output_index.transpose())
                    # EOS_index = np.zeros(FLAGS.beam_size)
                    # for beam in range(FLAGS.beam_size):
                    #     for char in range(len(target_output_index[beam])):
                    #         if target_output_index[beam][char] == word2index['<EOS>']:
                    #             EOS_index[beam] = char
                    #             break
                    # longest_beam_index = np.argmax(EOS_index)
                    # # predict_list = random.sample(target_output_index, 1)
                    # predict_list = target_output_index[longest_beam_index]
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
                target_output = target_output.replace('<UNK>', '')
                target_output = target_output.replace(' ', '')

                if (target_output == ''):
                    target_output = '我'
                # print (batch_source_inputs[index], batch_target_inputs[index], target_output)
                # print ("==================================================")
                # print ("Soruce Input: ", batch_source_inputs[index])
                # print ("Target Input: ", batch_target_inputs[index])
                # print ("Target Output: ", target_output)
                # print ("==================================================")
                target_outputs.append(target_output)
        
        # df = pd.DataFrame(np.array(source_inputs_list))
        # df.to_csv(input_testset_filename, index=False, header=False)
        df = pd.DataFrame(np.array(target_outputs))
        df.to_csv(output_testset_filename, index=False, header=False)

if __name__ == "__main__":
    inference()