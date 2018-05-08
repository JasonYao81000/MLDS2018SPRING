# build dict #
# python3 data_preprocessing.py ./filtered_clr_conversation.txt ./word_vocab.txt ./w2v_corpus.txt

import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io
# from gensim.models import Word2Vec



# Referenced from https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py.
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def build_dictionary(conversation_set, index_vocab, min_count, min_length):
    index2char = {}
    char2index = {}
    char_dictionary = []
    for index, vocab in index_vocab:
        index2char[int(index)] = vocab
        char2index[vocab] = int(index)
        char_dictionary.append(vocab)

    word_counts = {}
    senences_count = 0
    for sentences in conversation_set:
        for sentence in sentences:
            senences_count += 1
            for word in sentence.split(' '):
                if len(word) >= 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
    
    word_dictionary = [word for word in word_counts if word_counts[word] >= min_count]
    print ('Filtered words from %d to %d with min_count [%d]' % (len(word_counts), len(word_dictionary), min_count))
    
    print ('char [%d] word [%d]' % (len(char_dictionary), len(word_dictionary)))
    word_dictionary = char_dictionary + word_dictionary
    print ('word dictionary [%d]' % (len(word_dictionary)))
    
    index2word = {}
    # index2word[0] = '<pad>'
    # index2word[1] = '<bos>'
    # index2word[2] = '<eos>'
    # index2word[3] = '<unk>'

    word2index = {}
    # word2index['<pad>'] = 0
    # word2index['<bos>'] = 1
    # word2index['<eos>'] = 2
    # word2index['<unk>'] = 3

    for index, word in enumerate(word_dictionary):
        word2index[word] = index
        index2word[index] = word

    # word_counts['<pad>'] = senences_count
    # word_counts['<bos>'] = senences_count
    # word_counts['<eos>'] = senences_count
    # word_counts['<unk>'] = senences_count

    # conversation_pair = []
    # for sentence in conversation_set:
    #     for index in range(len(sentence) - 1):
    #         if len(sentence[index].split(' ')) >= min_length \
    #             and len(sentence[index + 1].split(' ')) >= min_length:
    #             conversation_pair.append((sentence[index], sentence[index + 1]))
    # print('Total training pairs are %d with min_length [%d].' % (len(conversation_pair), min_length))
 
    return word2index, index2word, word_dictionary

def build_train_pairs(conversation_set, min_length, max_decoder_steps):
    conversation_pair = []
    w2v_corpus = []
    for sentence in conversation_set:
        for index in range(len(sentence) - 1):
            if len(sentence[index].split(' ')) >= min_length \
                and len(sentence[index + 1].split(' ')) >= min_length:
                pad_source_sentence = ''
                pad_target_sentence = ''
                if len(sentence[index].split(' ')) < max_decoder_steps:
                    words = sentence[index].split(' ')
                    for i in range(max_decoder_steps):
                        if i < len(words):
                            pad_source_sentence = pad_source_sentence + words[i] + ' '
                        else:
                            pad_source_sentence = '<PAD> ' + pad_source_sentence
                else:
                    pad_source_sentence = sentence[index]
                
                if len(sentence[index + 1].split(' ')) < max_decoder_steps:
                    words = sentence[index + 1].split(' ')
                    for i in range(max_decoder_steps):
                        if i < len(words):
                            pad_target_sentence = pad_target_sentence + words[i] + ' '
                        else:
                            pad_target_sentence = pad_target_sentence + '<PAD>' + ' '
                else:
                    pad_target_sentence = sentence[index + 1]
                conversation_pair.append((pad_source_sentence.rstrip(), pad_target_sentence.rstrip()))
                
    print('Total training pairs are %d with min_length [%d].' % (len(conversation_pair), min_length))
    return conversation_pair

def build_w2v_corpus(conversation_set, max_decoder_steps):
    w2v_corpus = []
    for sentences in conversation_set:
        for sentence in sentences:
            words = sentence.split(' ')
            sentence = sentence + ' '
            if len(words) < max_decoder_steps:
                for i in range(max_decoder_steps):
                    if i >= len(words):
                        sentence = sentence + '<PAD> '
            w2v_corpus.append(sentence)

    return w2v_corpus

# def filter_token(string, filter_words):
#     # Characters filters.
#     # filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n，、。‧．；：？！'
#     # filters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'
#     words = string.split(' ')
#     for index, word in enumerate(words):
#         if word not in filter_words:
#             words[index] = words[index].replace(word, '<UNK>')
#     string = ' '.join(words)
#     return string

if __name__ == "__main__":
    np.random.seed(9487)

    min_count = 10
    min_length = 1
    max_decoder_steps = 15

    conversation_data = sys.argv[1] 
    word_vocab_file = sys.argv[2]    
    w2v_corpus_file = sys.argv[3]    

    filter_words = []
    with open(word_vocab_file, 'r') as f:
        for line in f:
            filter_word = line.split(' ')[1].replace('\n', '')
            filter_words.append(filter_word)

    conversation_set = []
    conversation = []
    with open(conversation_data, 'r') as f:
        for line in f:
            line = line.replace('\n', '').rstrip()
            if line == '+++$+++' and len(conversation) > 0:
                conversation_set.append(conversation)
                conversation = []
            else:
                if line != '+++$+++':
                    conversation.append(line)
            
    print('Total conversation set are %d.' % (len(conversation_set)))

    conversation_pair = build_train_pairs(conversation_set, min_length, max_decoder_steps)
    print(conversation_pair[0])
    print(conversation_pair[len(conversation_pair)-1])

    w2v_corpus = build_w2v_corpus(conversation_set, max_decoder_steps)
    print(w2v_corpus[0])
    print(w2v_corpus[len(w2v_corpus)-1])

    # index_vocab = []
    # with open(vocab_file, 'r') as f:
    #     for line in f:
    #         line = line.replace('\n', '')
    #         index_vocab.append(line.split(' '))

    # word2index, index2word, dictionary, conversation_pair = build_dictionary(conversation_set, index_vocab, min_count=min_count, min_length=min_length)

    # print(conversation_pair[0])
    # print(conversation_pair[len(conversation_pair)-1])
    
    max_sentence_length = max([len(x[0].split(' ')) for x in conversation_pair])
    avg_sentence_length = np.mean([len(x[0].split(' ')) for x in conversation_pair])

    print("Max. length of sentence: ", max_sentence_length)
    print("Avg. length of sentence: ", avg_sentence_length)

    # pickle.dump(word2index, open('./word2index.obj', 'wb'))
    # pickle.dump(index2word, open('./index2word.obj', 'wb'))
    # pickle.dump(dictionary, open('./dictionary.obj', 'wb'))
    pickle.dump(conversation_pair, open('./conversation_pair.obj', 'wb'))

    with open(w2v_corpus_file, 'w') as f:
        for sentence in w2v_corpus:
            f.write("%s\n" % (sentence))
    # with open(word_vocab_file, 'w') as f:
    #     for index, word in enumerate(index2word):
    #         f.write("%d %s\n" %(index, index2word[index]))
