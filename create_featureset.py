## Give each word unique id
## How to deal with different length vectors????
## Create feature sets
                                                                                                           
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
from Sentence import Sentence
hm_lines = 1000000
lemmatizer = WordNetLemmatizer()

def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r', buffering=100000) as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                sent = Sentence(l).build_sentence(l)
                all_words = word_tokenize(sent)
                lexicon += list(all_words)

    ## Counter creates dictionary like object with words as keys, and word_counts as values
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
    ## Play around with these parameters, idea is we dont care about dna sequences that appear all the time everywhere, or hardly at all                                                                                                                     

        l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    ## hot array corresponding to words in sentence, plus the classification of that word
    '''
    [
    [1 0 0 1 0 0 2], [1 0]
    [0 0 0 4 1 0 0], [0 1]
    ] 
    '''
    feature_set = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[: hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            feature_set.append([features, classification])
    return feature_set


def create_feature_sets_and_labels(pos, neg, outf, test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('small_pos.txt', lexicon, [1,0])
    features += sample_handling('small_neg.txt', lexicon, [0,1])
    ## if we dont shuffle, we train our network to classify all the positive reads first
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size*len(features))
    ## syntactic sugar for saying all of the 0'th elemnts of an nd arary, aka all our features
    train_x = np.array(features[:,0][:-testing_size])
    train_y = np.array(features[:,1][:-testing_size])

    test_x = np.array(features[:,0][-testing_size:])
    test_y = np.array(features[:,1][-testing_size:])

    ## return train_x, train_y, test_x, test_y

    outfile = open(outf, 'a')
    for feature in features:
        outfile.write(feature)
        print(feature)
    return features

if __name__ == '__main__':
    '''train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
    '''

    features = create_feature_sets_and_labels('small_pos.txt', 'small_neg.txt', 'outfile.txt')
        
