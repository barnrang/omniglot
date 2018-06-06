import tensorflow as tf

def slice_tensor_and_sum(x, way=20):
    sliced = tf.split(x, num_or_size_splits=way,axis=0)
    return tf.reduce_mean(sliced, axis=1)
