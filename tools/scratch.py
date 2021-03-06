import tensorflow as tf
from Sentence import Sentence
import numpy as np
'''
input > wieght > hidden layer 1 > activation function > weights > hidden layer 2 > activation function 2 > weights > output

compare output to intended output > cost function (cross entropy)
optimizzation function > minimize cost (AdamOPtimizer, SGD)

backpropogation to manipulate weights

feed forward + backprop = 1 epoch
'''


'''
one-hot encoding useful for multiclass classification
'''

## import exampel tensorflow data set of hand written digits
##from tensorflow.examples.tutorials.mnist import input_data

##mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from create_featureset import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('small_pos.txt','small_neg.txt', 'out_test.txt')



### Tensorflow can actually derive number of classes, but if we know how many classes we should have, it will sspeed up process
n_classes = 2

'''cant load all of data into memory at once for most sets, so we gotta use batches'''

batch_size = 128
total_batches = int(100000/batch_size)
hm_epochs = 10

##  height by width, but we flatten the 28*28 matrix
##  We also dont need to define dimmesnsions, if we do and pass in invalid dimmensions, tensorflow will throw error
##  else, we can handle matrices fo unknown sizes
x = tf.placeholder('float',[None, len(train_x[0])])
y = tf.placeholder('float')

def init_process(fin, fout):
    outfile = open(fout,'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split
        except:
            pass


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    ## weights dictionary, 5x5 convolution, 1 input, 32 feature
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_confc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024,n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_confc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    ## reshpe input to be 2d
    x = tf.reshape(x, shape=[-1,28,28,1])

    conv1 = conv2d(x, weights['W_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = conv2d(conv1, weights['W_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmaul(fc, weights['W_fc'])+biases['b_fc'])

    output = tf.matmul(fc, weights['out']+biases['out'])
    
    



    biases = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}


    return output
    ## finish decleration of computational tensor graph for our NN model


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    ##cross entropy cost function to calculate difference between our prediciton, and the known correct output
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )


    ## use Adam optimization funciton, a first-order gradient-based optimzation method for stochastic objective funcitons
    ## We want to minimize the difference between our prediciton and actual value
    ## default learning rate - 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    ## begin session
    ## NOTE: using 'with' syntax will automatically close the sessions when were done 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
        except:
            epoch = 1
        

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, 'model.ckpt')
            epoch_loss = 1
            with open('lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=20000) as f:
                batch_x = []
                batch_y = []
                batches_run = 0
            for line in f:
                label = line.split(':::')[0]
        ## Beging traning data
        '''for epoch in range(hm_epochs):
            epoch_loss = 0
            ## '_' short hand for variable we dont actually care about
            
            i = 0
            while i <  len(train_x):
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                ##run through our data to minimize our cost with optimizer function which will modify weights in our layer
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y:batch_y})
                epoch_loss += c
                i += batch_size
            print("Epoch", epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
           '''
            

        ## calculate all accuracies and evaluate accuracy of our model
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)

## Give each word unique id
## How to deal with different length vectors????
## Create feature sets        
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter

hm_lines = 1000000

def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            conents = f.readlines()
            for l in contents[:hm_lines]:
                sent = Sentence(l).build_sentence(l)
                all_words = word_tokenize(s)
                lexicon += list(all_words)

    ## Counter creates dictionary like object with words as keys, and word_counts as values
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        ## Play around with these parameters, idea is we dont care about dna sequences that appear all the time everywhere, or hardly at all
        if 10000 > w_counts[w] > 2:
            l2.append(w)

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
