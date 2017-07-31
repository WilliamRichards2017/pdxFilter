import numpy as np
import tensorflow as tf
from sklearn import preprocessing, cross_validation, neighbors, svm
import preprocess
from tensorflow.contrib import learn

tf.flags.DEFINE_float("dev_sample_percentage", .9, "Percentage of traning data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "pos.txt", "Data source for positive data (human reads)")
tf.flags.DEFINE_string("negative_data_file", "neg.txt", "Data source for negative data (mouse reads)")


FLAGS = tf.flags.FLAGS
## Load data                                                                                                                                                                                                                         
x_text, y = preprocess.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

## Build up vocab                                                                                                                                                                                                                  
 ## TODO: altar this and use on-hot encoding                                                                                                                                                                                          
max_document_length = max([len(x.split(" ")) for x in x_text])

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

## use numpy to perform 1hot encoding                                                                                                                                                                                                 
x = np.array(list(vocab_processor.fit_transform(x_text)))
##enc = OneHotEncoder()                                                                                                                                                                                                               
##enc.fit(x)                                                                                                                                                                                                                          
##enc.transform(x).toarray()                                                                                                                                                                                                          

## Shuffle data                                                                                                                                                                                                                       
np.random.seed(421)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

## since our data set is large, it is not computationally worth it to use k-fold cross-validation                                                                                                                                     
## rather, we split our data randomly into train and test sets                                                                                                                                                                        
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage*float(len(y)))
x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

x_train.reshape(8002, 1)
y_train.reshape(8002, 1)



print(x_train)
print(x_test)

clf = svm.SVC()
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test)
print(acc)

##prediction = clf.predict(x_train, y_train)
##print(prediction)
