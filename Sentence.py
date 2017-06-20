import tensorflow as tf
import numpy as np
from itertools import *
from string import ascii_lowercase


class Sentence:
    alphabet = ['a','c','g','t']
    dictionary = list(product(alphabet, repeat=3))
    def __init__(self, read):
        self.read = read
        self.sentece

    def build_sentence(read):
        sentence = []
        for x in range(0, len(read)-3):
            word = read[x] + read[x+1] + read[x+2]
            sentence.append(word)
        return sentence

for d in Sentence.dictionary:
    print(d)

sent = Sentence.build_sentence("acgtacgtacgtacgt")

for s in sent:
    print(s)

a = np.zeros(shape=(len(sent),len(Sentence.dictionary)))

for i in range(0,len(sent)):
    for j in range(0,len(Sentence.dictionary)):
        print("comparing", sent[i], "and", ''.join(Sentence.dictionary[j]), "\n")
        if sent[i] == ''.join(Sentence.dictionary[j]):
            print("its a match bb")
            a[i][j] = 1

print(np.matrix(a))
