##convolutional neural network 
import tensorflow as tf
import pandas as pd

class cnn(object):

    ## Network definition
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.sequence_length=sequence_length
        self.num_classes=num_classes
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        
        ## define embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
        ## define convolutional layers
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1,1,1,1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                ## Max pooling
                pooled = tf.nn.max_pool(h, ksize=[1,sequence_length - filter_size + 1,1,1], strides=[1,1,1,1],padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                
        ## combine all features into one big vector        
        num_filters_total = num_filters *   len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        ## droupout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
          ## define output  
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            ##self.confidence = tf.concat([tf.self.scores,1), tf.concat([tf.cast(self.predictions, tf.float32), self.input_y],2)],2)
            temp = tf.concat([self.scores, self.input_y],1)
            self.confidence = tf.concat([temp, tf.expand_dims(tf.cast(self.predictions, tf.float32),1)],1)
            
            



        # calculate entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda*l2_loss
            
        ## calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


            
            
