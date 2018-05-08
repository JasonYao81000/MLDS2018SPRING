# build dict #
# python3 build_filter_words.py ./clr_conversation.txt ./vocab.txt ./word_vocab.txt

import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io
# from gensim.models import Word2Vec

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
    word2index = {}

    for index, word in enumerate(word_dictionary):
        word2index[word] = index
        index2word[index] = word
    
    return word2index, index2word, word_dictionary

if __name__ == "__main__":
    np.random.seed(9487)

    min_count = 20
    min_length = 2

    conversation_data = sys.argv[1] 
    vocab_file = sys.argv[2]    
    word_vocab_file = sys.argv[3]    

    conversation_set = []
    conversation = []
    with open(conversation_data, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line == '+++$+++' and len(conversation) > 0:
                conversation_set.append(conversation)
                conversation = []
            else:
                if line != '+++$+++':
                    conversation.append(line)

    print('Total conversation set are %d.' % (len(conversation_set)))

    index_vocab = []
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            index_vocab.append(line.split(' '))

    word2index, index2word, dictionary = build_dictionary(conversation_set, index_vocab, min_count=min_count, min_length=min_length)

    with open(word_vocab_file, 'w') as f:
        for index, word in enumerate(index2word):
            f.write("%d %s\n" %(index, index2word[index]))
