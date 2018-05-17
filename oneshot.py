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

BASE_PATH = "python/one-shot-classification/all_runs"

def load_model():
    pass

def load_data(test=0):
    train_batch = []
    test_batch = []
    base_dir = os.path.join (BASE_PATH, f"run{%2d}".(test))
    train_dir = os.listdir(os.path.join(base_dir, 'training'))
    test_dir = os.listdir(os.path.join(base_dir, 'test'))
    for i in range(len(train_dir)):
        train_batch.append(plt.imread(image_dir[i]).astype(np.uint8))
        test_batch.append(plt.imread(test_dir[i]).astype(np.uint8))

    return train_batch, test_batch

def load_label(test=0):
    text_file = os.path.join(BASE_PATH, f"run{%2d}/class_labels".(test))
    f = open(text_file, 'r')
    pair = []
    for line in f:
        path1, path2 = line.split(' ')
        idx1, idx2 = int(path1[-6:-4]), int(path2[-6:-4])
        pair.append((idx1,idx2))
    return pair

def retrieve_feature(model, train_batch, test_batch):
    train_feature = model.predict(train_batch)
    test_feature = model.predict(test_batch)


if __name__ == "__main__":
    pass
