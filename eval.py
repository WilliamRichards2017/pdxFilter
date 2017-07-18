import tensorflow as tf
import numpy as np
import os
import time
import datetime
import preprocess
from cnn import cnn
from tensorflow.contrib import learn
import csv

tf.flags.DEFINE_string("positive_data_file", "small_pos.txt", "Data source for human reads." )
tf.flags.DEFINE_string("negative_data_file", "small_neg.txt", "Data source for mice reads.")

tf.flags.DEFINE_string("sample_positive_data_file", "sample_pos.txt", "Sample human reads for evaluation." )
tf.flags.DEFINE_string("sample_negative_data_file", "sample_neg.txt", "Sample mouse reads for evaluation.")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size: (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/uufs/chpc.utah.edu/common/home/u0401321/classifier/runs/1499288992/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of operations on  device")

## Parse flag params
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


## load in data to eval on trained model
if FLAGS.eval_train:
    x_raw, y_test = preprocess.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw, y_test = preprocess.load_data_and_labels(FLAGS.sample_positive_data_file, FLAGS.sample_negative_data_file)
    y_test = np.argmax(y_test, axis=1)

##vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
##vocab_path = '~/classifier/vocab'
##vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
##x_test = np.array(list(vocab_processor.transform(x_raw)))

max_document_length = max([len(x.split(" ")) for x in x_raw])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_test = np.array(list(vocab_processor.fit_transform(x_raw)))


validation_metrics = {
     "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    y_test[0],
    y_test[1],
    every_n_steps=50,
    metrics=validation_metrics)

print("\nEvaluating...\n")

start_time = time.time()

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob=graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        batches = preprocess.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        all_predictions=[]
        
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


if y_test  is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictioncs.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

print("\nTime spent evaluating: {0:.3f} min.".format((time.time() - start_time) / float(60)))


