import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import transform

DIMENSION = 105
CHANNEL = 1

def prob(x):
    return tf.less(tf.random_uniform([1]),x)[0]

def random_erase(x):
    pack = tf.random_uniform([2], minval=15, maxval=40, dtype=tf.int32)
    width, height = pack[0], pack[1]
    wst = tf.random_uniform([1], minval=0, maxval=DIMENSION - width, dtype = tf.int32)[0]
    hst = tf.random_uniform([1], minval=0, maxval=DIMENSION - height, dtype = tf.int32)[0]
    erase = tf.random_uniform([width, height, CHANNEL])
    padding = tf.convert_to_tensor([[wst, DIMENSION - wst - width],[hst, DIMENSION - hst - height],[0,0]])
    padded = tf.pad(erase, padding, constant_values = 1.)
    return x * padded

def rotate(x):
    theta = tf.random_uniform([1],-10,10)[0]/(180)*np.pi
    sub_tran = tf.convert_to_tensor([[tf.cos(theta),tf.sin(theta)],[-tf.sin(theta), tf.cos(theta)]])
    return tf.matmul(x, sub_tran)

def shear(x):
    p = tf.random_uniform([2],-0.1,0.1)
    sub_tran = tf.convert_to_tensor([[1,p[1]],[p[0], 1]])
    return tf.matmul(x, sub_tran)

def scale(x):
    s = tf.random_uniform([2],0.8,1.2)
    sub_tran = tf.convert_to_tensor([[s[0],0],[0,s[1]]])
    return tf.matmul(x, sub_tran)

def translation():
    return tf.random_uniform([2],-1,1)

def affine_transform(X, rate):
    trans_matrix = tf.eye(2)
    trans_matrix = tf.cond(prob(rate),lambda: rotate(trans_matrix), lambda: trans_matrix)
    trans_matrix = tf.cond(prob(rate),lambda: shear(trans_matrix), lambda: trans_matrix)
    trans_matrix = tf.cond(prob(rate),lambda: scale(trans_matrix), lambda: trans_matrix)
    X = tf.cond(prob(rate),lambda: tf.map_fn(random_erase, X), lambda: X)
    t = tf.cond(prob(rate), translation, lambda: tf.zeros(2))
    a0,a1,b0,b1 = trans_matrix[0][0],trans_matrix[0][1],trans_matrix[1][0],trans_matrix[1][1]
    a2,b2 = t[0],t[1]
    return transform(X, [a0,a1,a2,b0,b1,b2,0,0])

def transform_gate(X, rate, is_training):
    output = tf.cond(is_training, lambda: affine_transform(X, rate), lambda: X)
    return X
