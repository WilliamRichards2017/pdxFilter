import numpy as np
import itertools
from Sentence import Sentence

def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples =list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            suffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, datasize)
            yield shuffled_data[start_index:end_index]
