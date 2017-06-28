import tensorflow as tf

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
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

## define number of nodes in our hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

### Tensorflow can actually derive number of classes, but if we know how many classes we should have, it will sspeed up process
n_classes = 10

'''cant load all of data into memory at once for most sets, so we gotta use batches'''

batch_size = 100

##  height by width, but we flatten the 28*28 matrix
##  We also dont need to define dimmesnsions, if we do and pass in invalid dimmensions, tensorflow will throw error
##  else, we can handle matrices fo unknown sizes
x = tf.placeholder('=float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    ## define init weights as random tensor of shape equal to image size by number of nodes in first hidden layer
    ## bias doesnt need a shape, so we just create a random bias for each of our nodes

    ## (input_data * weights) + biases

    ## We need biases to overcome the problem of inputs of 0, so neurons can fire even if inputs are 0
    
    hidden_1_layer = {'wieghts':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'baises':tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_2_layer = {'wieghts':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'baises':tf.Variable(tf.random_normal(n_nodes_hl2))}

    hidden_3_layer = {'wieghts':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'baises':tf.Variable(tf.random_normal(n_nodes_hl3))}

    output_layer = {'wieghts':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'baises':tf.Variable(tf.random_normal(n_classes))}

    ## matrix multiplication of input_data and weights, plus biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights'])+hidden_1_layer['biases'])
    ## apply rectifiedlinear activation function to our output of layer1
    l1 = tf.nn.relu(l1)

    ##reapeat matrix multiplacation and activation function for each hidden layer
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights'])+hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights'])+hidden_3_layer['biases'])
    l3 = tf.nn.relu(l1)

    output_layer = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output
    ## finish decleration of computational tensor graph for our NN model



    
