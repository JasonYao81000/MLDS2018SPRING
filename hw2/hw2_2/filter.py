# build dict #
# python3 filter.py ./clr_conversation.txt ./word_vocab.txt ./filtered_clr_conversation.txt

import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io
# from gensim.models import Word2Vec

def filter_token(string, filter_words):
    words = string.split(' ')
    for index, word in enumerate(words):
        if word not in filter_words:
            words[index] = words[index].replace(word, '<UNK>')
    string = ' '.join(words)
    return string

if __name__ == "__main__":
    np.random.seed(9487)

    min_count = 10
    min_length = 1
    max_decoder_steps = 15

    conversation_data = sys.argv[1] 
    word_vocab_file = sys.argv[2]    
    filtered_conversation_data = sys.argv[3]    

    filter_words = []
    with open(word_vocab_file, 'r') as f:
        for line in f:
            filter_word = line.split(' ')[1].replace('\n', '')
            filter_words.append(filter_word)

    conversation = []
    line_number = 0
    with open(conversation_data, 'r') as f:
        for line in f:
            line = line.replace('\n', '').rstrip()
            if line == '+++$+++' and len(conversation) > 0:
                conversation.append(line)
            else:
                if line != '+++$+++':
                    line = filter_token(line, filter_words)
                    line = '<BOS> ' + line
                    words = line.split(' ')
                    if len(words) <= max_decoder_steps - 1:
                        line = line + ' <EOS>' 
                    else:
                        new_line = ''
                        for i in range(max_decoder_steps - 1):
                            new_line = new_line + words[i] + ' '
                        line = new_line + '<EOS>'
                    conversation.append(line)
            line_number = line_number + 1
            print('number of line: %d' % (line_number), end='\r')
            
    with open(filtered_conversation_data, 'w') as f:
        for sentence in conversation:
            f.write("%s\n" % (sentence))
