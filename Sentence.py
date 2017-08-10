import tensorflow as tf
import numpy as np
from itertools import *
from string import ascii_lowercase


class Sentence:
    alphabet = ['a','c','g','t', 'n']
    dictionary = list(product(alphabet, repeat=3))
    def __init__(self, read):
        self.read = read
        

    def build_sentence(self, read):
        sentence = ""
        for x in range(0, len(read)-3):
            word = read[x:x+3]
            sentence += word + " "
        
        for x in range(0, len(read)-10):
            word = read[x:x+10]
            sentence += word + " "

        '''for x in range(0, len(read)-25):
            word = read[x:x+25]
            sentence += word + " "
        '''
        return sentence

    def build_input_matrix(self, read):
        sent = self.build_sentence(read)
        a = np.zeros(shape=(len(sent),len(Sentence.dictionary)))

        for i in range(0,len(sent)):
            for j in range(0,len(Sentence.dictionary)):
                ##print("comparing", sent[i], "and", ''.join(Sentence.dictionary[j]), "\n")
                if sent[i] == ''.join(Sentence.dictionary[j]):
                    a[i][j] = 1
        return a




