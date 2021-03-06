import tensorflow as tf
import numpy as np
from Sentence import Sentence
import preprocess
from cnn import cnn
import time
import datetime
import os
from tensorflow.contrib import learn
from sklearn.preprocessing import OneHotEncoder

start_time = time.time()

tf.flags.DEFINE_float("dev_sample_percentage", .01, "Percentage of traning data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "pos.txt", "Data source for positive data (human reads)")
tf.flags.DEFINE_string("negative_data_file", "neg.txt", "Data source for negative data (mouse reads)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 125, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,5,10", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# parse flags
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")

## Load data
x_text, y, ids = preprocess.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

## Build up vocab
## TODO: altar this and use on-hot encoding
max_document_length = max([len(x.split(" ")) for x in x_text])

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x = np.array(list(vocab_processor.fit_transform(x_text)))

## Shuffle data
np.random.seed(420)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
id_shuffled = ids[shuffle_indices]

## since our data set is large, it is not computationally worth it to use k-fold cross-validation
## rather, we split our data randomly into train and test sets
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage*float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
id_train, id_dev = id_shuffled[:dev_sample_index], id_shuffled[dev_sample_index:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


## Start training our model
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    ## using with will automatically release memory when session is over
    with sess.as_default():
        tcnn = cnn(
            sequence_length=x_train.shape[1], 
            num_classes=y_train.shape[1], 
            vocab_size=len(vocab_processor.vocabulary_), 
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        
        ## define training proceucur
        global_step = tf.Variable(0, name="global_step",trainable=False)
        # Use Adam optimization, a first-order gradient-based optimazation method for stochastic objective functions 
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(tcnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # calculate gradient and sparcity
        grad_summaries = []
        for g,v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            ## Create directory to write out summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            
            ## loss and accuracy summaries
            loss_summary = tf.summary.scalar("loss", tcnn.loss)
            acc_summary = tf.summary.scalar("accuracy", tcnn.accuracy)
                        
            # train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            

            # dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint directory (needed for tensorflow to run)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            print("checkpoints are at{}".format(checkpoint_dir))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # init variables
            sess.run(tf.global_variables_initializer())
            
            #Perform a single training step on batch of data
            def train_step(x_batch, y_batch, id_batch):
                feed_dict = {
                    tcnn.input_x: x_batch,
                    tcnn.input_y: y_batch,
                    tcnn.input_id: id_batch,
                    tcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                ## Run tensorflow session to calculate desired information of our batch
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, tcnn.loss, tcnn.accuracy],
                    feed_dict
                )
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc{:g}".format(time_str,step,loss,accuracy))
                train_summary_writer.add_summary(summaries,step)

                ## evaluate model on dev set
            def dev_step(x_batch, y_batch, id_batch, writer=None):

                feed_dict = {
                    tcnn.input_x: x_batch,
                    tcnn.input_y: y_batch,
                    tcnn.input_id: id_batch,
                    tcnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, confidence = sess.run(
                    [global_step, dev_summary_op, tcnn.loss, tcnn.accuracy, tcnn.confidence],
                    feed_dict
                )
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc{:g}".format(time_str,step,loss,accuracy))
                np.savetxt( 'confidence.csv', confidence, delimiter=',', fmt='%.5e')
                if writer:
                    writer.add_summary(summaries, step)
                
            
            ## generate batches from preprocess.py
            batches = preprocess.batch_iter(list(zip(x_train, y_train, id_train)), FLAGS.batch_size, FLAGS.num_epochs)

            for batch in batches:
                x_batch, y_batch, id_batch = zip(*batch)
                train_step(x_batch, y_batch, id_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every==0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, id_dev, writer=dev_summary_writer)
                    print("\nTime spent sampling pos reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))

                    print("")
                if current_step % FLAGS.checkpoint_every ==0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("saved modelcheckpoint to {}\n".format(path))
