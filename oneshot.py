from keras.utils import np_utils
from keras import callbacks as cb
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import *
from keras.models import Sequential
from keras import regularizers as rg
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras import backend as K
import os

import numpy.random as rng

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from python.dataloader import loader
from kerasloader import DataGenerator
from model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from transform import transform_gate

BASE_PATH = "python/one-shot-classification"

def load_conv_model():
    pass

def load_data(test=1):
    train_batch = []
    test_batch = []
    base_dir = os.path.join (BASE_PATH, "run%02d" % (test))
    train_path = os.path.join(base_dir, 'training')
    test_path = os.path.join(base_dir, 'test')
    train_dir = os.listdir(train_path)
    test_dir = os.listdir(test_path)
    train_dir.sort()
    test_dir.sort()
    for i in range(len(train_dir)):
        train_batch.append(plt.imread(os.path.join(train_path, train_dir[i])).astype(np.uint8))
        test_batch.append(plt.imread(os.path.join(test_path, test_dir[i])).astype(np.uint8))

    train_batch = np.expand_dims(train_batch, axis=-1)
    test_batch = np.expand_dims(test_batch, axis=-1)
    return train_batch, test_batch

def load_label(test=1):
    text_file = os.path.join(BASE_PATH, "run%02d/class_labels.txt" % (test))
    f = open(text_file, 'r')
    pair = []
    for line in f:
        path1, path2 = line.split(' ')
        idx1, idx2 = int(path1[-6:-4]), int(path2[-7:-5])
        pair.append((idx1,idx2))
    return pair

def retrieve_feature(model, train_batch, test_batch):
    '''
    Input 10 - train_batch 10 - test_batch
    Out distance: [N_train, N_test]
    '''
    train_feature = model.predict(train_batch)
    test_feature = model.predict(test_batch)
    train_reshape = np.expand_dims(train_feature, axis=1)
    test_reshape = np.expand_dims(test_feature, axis=0)
    dist = np.sum(np.abs(train_reshape - test_reshape), axis=-1)
    return dist

def cal_acc(dist, label):
    pred = np.argmin(dist, axis=0) + 1
    target = [x[1] for x in label]
    return np.mean(pred == target)

if __name__ == "__main__":
    acc = []
    N_set = 20
    model = load_conv_model()
    for i in range(N_set):
        train_batch, test_batch = load_data(i)
        pair = load_label(i)
        dist = retrieve_feature(model, train_batch, test_batch)
        acc.append(cal_acc(dist))
    print(np.mean(acc))
