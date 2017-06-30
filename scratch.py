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
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    ## define init weights as random tensor of shape equal to image size by number of nodes in first hidden layer
    ## bias doesnt need a shape, so we just create a random bias for each of our nodes

    ## (input_data * weights) + biases

    ## We need biases to overcome the problem of inputs of 0, so neurons can fire even if inputs are 0
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    ## matrix multiplication of input_data and weights, plus biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    ## apply rectifiedlinear activation function to our output of layer1
    l1 = tf.nn.relu(l1)

    ##reapeat matrix multiplacation and activation function for each hidden layer
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l1)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output
    ## finish decleration of computational tensor graph for our NN model


def train_neural_network(x):
    prediction = neural_network_model(x)
    ##cross entropy cost function to calculate difference between our prediciton, and the known correct output
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )


    ## use Adam optimization funciton, a first-order gradient-based optimzation method for stochastic objective funcitons
    ## We want to minimize the difference between our prediciton and actual value
    ## default learning rate - 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    ## begin session
    ## NOTE: using 'with' syntax will automatically close the sessions when were done 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        ## Beging traning data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            ## '_' short hand for variable we dont actually care about
            for _ in range(int(mnist.train.num_examples/batch_size)):
                ## use batches to avoid loading all data into memory
                x_epoch,y_epoch = mnist.train.next_batch(batch_size)
                ##run through our data to minimize our cost with optimizer function which will modify weights in our layer
                _, c = sess.run([optimizer, cost], feed_dict={x: x_epoch, y:y_epoch})
                epoch_loss += c
            print("Epoch", epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        ## calculate all accuracies and evaluate accuracy of our model
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
        
