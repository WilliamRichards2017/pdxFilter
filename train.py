import tensorflow as tf
import numpy as np
from Sentence import Sentence
import time

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of traning data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "sequences.txt", "Data source for positive data (human reads)")

tf.flags.DEFINE_string("negative_data_file", "neg_sequences.txt", "Data source for \
negative data (mouse reads)")

pos = 'sequences.txt'
neg = 'neg_sequences.txt'


start_time = time.time()
with open(pos) as f:
    for line in f:
        s = Sentence(line)
        a = s.build_input_matrix(line)

print("\nTime spent building input matrices: {0:.3f} min.".format((time.time() - start_time) \
/ float(60)))
