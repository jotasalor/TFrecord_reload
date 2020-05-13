import tensorflow as tf

path = 'D:/TFRecord/bdd100k_train.record-00000-of-00100'
for example in tf.python_io.tf_record_iterator(path):
    result = tf.train.Example.FromString(example)
