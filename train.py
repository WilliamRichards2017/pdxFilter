import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of traning data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "sequences.txt", "Data source for positive data (human reads)")
