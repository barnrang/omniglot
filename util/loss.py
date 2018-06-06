import tensorflow as tf

def prior_dist(feature, pred):
    pred_dist = tf.reduce_sum(pred ** 2, axis=1, keep_dims=True)
    feature_dist = tf.reduce_sum(feature ** 2, axis=1, keep_dims=True)
    dot = tf.matmul(pred, tf.transpose(feature))
    return pred + tf.transpose(feature_dist) - 2 * dot

#def prior_loss(target, pred):
#    return tf.losses.softmax_cross_entropy(target, -pred)
#
#def prior_acc(target, pred):
#

