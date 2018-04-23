# build dict #
# python3 data_preprocessing.py ./MLDS_hw2_1_data/training_data/feat/ ./MLDS_hw2_1_data/training_label.json

import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io
# from gensim.models import Word2Vec

max_decoder_steps = 15

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

def build_dictionary(sentences, min_count):
    word_counts = {}
    senences_count = 0
    for sentence in sentences:
        senences_count += 1
        for word in sentence.lower().split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1
    
    dictionary = [word for word in word_counts if word_counts[word] >= min_count]
    print ('Filtered words from %d to %d with min_count [%d]' % (len(word_counts), len(dictionary), min_count))

    index2word = {}
    index2word[0] = '<pad>'
    index2word[1] = '<bos>'
    index2word[2] = '<eos>'
    index2word[3] = '<unk>'

    word2index = {}
    word2index['<pad>'] = 0
    word2index['<bos>'] = 1
    word2index['<eos>'] = 2
    word2index['<unk>'] = 3

    for index, word in enumerate(dictionary):
        word2index[word] = index + 4
        index2word[index + 4] = word

    word_counts['<pad>'] = senences_count
    word_counts['<bos>'] = senences_count
    word_counts['<eos>'] = senences_count
    word_counts['<unk>'] = senences_count

    # # Initial vector for embedding layer's bias.
    # embedding_bias_init_vector = np.array([1.0 * word_counts[index2word[index]] for index in index2word])
    # embedding_bias_init_vector /= np.sum(embedding_bias_init_vector) # normalize to frequencies
    # embedding_bias_init_vector = np.log(embedding_bias_init_vector)
    # embedding_bias_init_vector -= np.max(embedding_bias_init_vector) # shift to nice numeric range
    
    return word2index, index2word, dictionary

def filter_token(string):
    # Characters filters.
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    for c in filters:
        string = string.replace(c, '')
    return string

if __name__ == "__main__":
    np.random.seed(9487)

    video_feat_folder = sys.argv[1]
    training_label_json_file = sys.argv[2]

    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    # Remove '.avi' from filename.
    video_IDs = [filename[:-4] for filename in video_feat_filenames]

    video_feat_dict = {}
    for filepath in video_feat_filepaths:
        video_feat = np.load(filepath)
        video_ID = filepath[: -4].replace(video_feat_folder, "")
        video_feat_dict[video_ID] = video_feat
    
    video_caption = json.load(open(training_label_json_file, 'r'))
    video_caption_dict={}
    captions_corpus = []
    for video in video_caption:
        filtered_captions = [filter_token(sentence) for sentence in video["caption"]]
        video_caption_dict[video["id"]] = filtered_captions
        captions_corpus += filtered_captions


    word2index, index2word, dictionary = build_dictionary(captions_corpus, min_count=3)
    
    pickle.dump(word2index, open('./word2index.obj', 'wb'))
    pickle.dump(index2word, open('./index2word.obj', 'wb'))
    # pickle.dump(embedding_bias_init_vector, open('./embedding_bias_init_vector.obj', 'wb'))

    ID_caption = []
    captions_words = []

    # corpus_txt_file = io.open('./corpus.txt', 'w')
    # corpus_txt_file.write('<pad> <pad> <pad> <pad> <pad> ' + '<unk> <unk> <unk> <unk> <unk>' + '\n')

    words_list = []
    for ID in video_IDs:
        for caption in video_caption_dict[ID]:
            # corpus_txt_file.write('<bos> ')
            # for word in caption.lower().split():
            #     if word in w2v_model.wv.vocab:
            #     if word in dictionary:
            #         corpus_txt_file.write(word + ' ')
            #     else:
            #         corpus_txt_file.write('<unk> ')
            # corpus_txt_file.write('<eos> ')
            # if (len(caption.split()) < max_decoder_steps):
            #     for _ in range(max_decoder_steps - len(caption.split()) - 1):
            #         corpus_txt_file.write('<pad> ')
            # # corpus_txt_file.write(caption.lower() + ' ')
            # corpus_txt_file.write(' \n')
            ID_caption.append((video_feat_dict[ID], caption))
            words = caption.split()
            captions_words.append(words)
            for word in words:
                words_list.append(word)

    # corpus_txt_file.close()

    captions_words_set = np.unique(words_list, return_counts=True)[0]
    max_captions_length = max([len(words) for words in captions_words])
    avg_captions_length = np.mean([len(words) for words in captions_words])
    num_unique_tokens_captions = len(captions_words_set)

    print("np.shape(ID_caption): ", np.shape(ID_caption))
    print("Max. length of captions: ", max_captions_length)
    print("Avg. length of captions: ", avg_captions_length)
    print("Number of unique tokens of captions: ", num_unique_tokens_captions)

    print("Shape of features of first video: ", ID_caption[0][0].shape)
    print("ID of first video: ", video_IDs[0])
    print("Caption of first video: ", ID_caption[0][1])

    # pickle.dump(ID_caption, open('ID_caption.obj', 'wb'))
    pickle.dump(video_IDs, open('video_IDs.obj', 'wb'))
    pickle.dump(video_caption_dict, open('video_caption_dict.obj', 'wb'))
    pickle.dump(video_feat_dict, open('video_feat_dict.obj', 'wb'))