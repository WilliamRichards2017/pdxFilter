import tensorflow as tf
import numpy as np
from Sentence import Sentence
import preprocess
import time

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of traning data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "pos.txt", "Data source for positive data (human reads)")
tf.flags.DEFINE_string("negative_data_file", "neg.txt", "Data source for \
negative data (mouse reads)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")


## Load data
x_text, y = preprocess.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

for x in x_text:
    print(x)

'''max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VOcabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))


## Shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

##Use kfold cross-validation in preprocess instead of this segment of code
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage*float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
'''




'''pos = 'sequences.txt'
neg = 'neg_sequences.txt'


start_time = time.time()
with open(pos) as f:
    for line in f:
        s = Sentence(line)
        a = s.build_input_matrix(line)

print("\nTime spent building input matrices: {0:.3f} min.".format((time.time() - start_time) \
/ float(60)))
'''
