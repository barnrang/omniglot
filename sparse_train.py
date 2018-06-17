
# # import comet_ml in the top of your file
# from comet_ml import Experiment

# # Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="qRngFGVIBc32hxbR60UCEvavQ", project_name="omniglot")

import tensorflow as tf

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


import numpy.random as rng

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from python.dataloader import loader
from priorloader import DataGenerator
from prior_model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from transform import transform_gate
from util.tensor_op import *
from util.loss import *
input_shape = (28,28,1)
batch_size = 20
way = 20
shot = 1  

if __name__ == "__main__":
    conv = conv_net()
    sample = tf.placeholder(tf.float32, [None, *(input_shape)])
    reset = tf.placeholder(tf.bool)
    out_feature = conv(sample)
    store_weight = tf.cond(reset,lambda: out_feature, lambda: store_weight + out_feature)
    inp = tf.placeholder(tf.float32, [None, *(input_shape)])
    map_feature = conv(inp)
    pred = Lambda(lambda x:prior_dist(x))([store_feature/shot, map_feature]) #negative distance
    combine = Model([sample, inp], pred)
    optimizer = Adam(0.0001)
    combine.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    print(combine.summary())
    train_loader = DataGenerator(way=way*3, shot=shot)
    val_loader = DataGenerator(data_type='val',way=way, shot=shot)
    save_model = cb.ModelCheckpoint('model/proto', monitor='val_loss',save_best_only=True)
    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)
    combine.fit_generator(train_loader,epochs=40,validation_data=val_loader,  use_multiprocessing=True, workers=4, shuffle=False)


# images, labels = zip(*list(loader('python/images_background')))
# images = np.expand_dims(images, axis=-1)
# images = np.repeat(images, repeats=3, axis=-1)
# print(images.shape)
# main_labels, sub_labels= [x[0] for x in labels], [x[1] for x in labels]
# encoder = LabelBinarizer()
# enc_main_labels = encoder.fit_transform(main_labels)
# output_num = len(np.unique(main_labels))
# bottleneck_model = conv_model()
# bottleneck_model.trainable = False
# inp = Input(shape=(105,105,3))
# features = bottleneck_model(inp)
# prediction = class_model(features)
# full_model = Model(inputs=inp, outputs=prediction)
# adam = Adam(1e-3)
# full_model.compile(optimizer=adam,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# full_model.fit(x=images, y=enc_main_labels, batch_size=32, epochs=100, validation_split=0.2)

# def class_model(inp):
#     x = Flatten()(inp)
#     x = BatchNormalization()(x)
#     x = Dense(256, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dense(output_num, activation='softmax')(x)
#     return x
