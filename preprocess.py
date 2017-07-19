import numpy as np
import itertools
import re
from Sentence import Sentence

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):
    print("Loading data and labels")
    p_examples = list(open(positive_data_file, "r", buffering=100000).readlines())
    p_examples = [s.strip() for s in p_examples]

    p_id = [s.split(',')[0] for s in p_examples]
    p_examples = [s.split(',')[1] for s in p_examples]

    positive_examples=[]
    for s in p_examples:
        sent = Sentence(s).build_sentence(s)
        positive_examples.append(Sentence(s).build_sentence(s))

    n_examples =list(open(negative_data_file, "r", buffering=100000).readlines())
    n_id = [s.split(',')[0] for s in n_examples]
    n_examples = [s.split(',')[1] for s in n_examples]
    for n in n_examples:
        print(n)
    ## n_examples = [s.strip() for s in n_examples]
    negative_examples=[]

    for n in n_examples:
        sent = Sentence(n).build_sentence(n)
        negative_examples.append(sent)
        
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    ids = np.concatenate([p_id, n_id],0)
    return [x_text, y, ids]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]

## Manual crossvalidation                                                                                                           
def kfold_fit(data_matrix, labels, classifier, n_folds=3):
    ## define kfold                                                                                                                  
    (num_rows, num_cols) = np.shape(data_matrix)
    kf = cv.KFold(num_rows, n_folds=n_folds)

    for train, test in kf: 

        data_train = data_matrix[train]
        data_test = data_matrix[test]
        labels_train = labels[train]
        labels_test = labels[test]

    classifier.fit(data_train, labels_train)
    return classifier
