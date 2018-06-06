
# import comet_ml in the top of your file
#from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
#experiment = Experiment(api_key="qRngFGVIBc32hxbR60UCEvavQ", project_name="omniglot")

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
import tensorflow as tf

import numpy.random as rng

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from python.dataloader import loader
from siameseloader import DataGenerator
from model import conv_net, hinge_loss, l2_distance, l1_distance
from transform import transform_gate

input_shape = (105,105,1)
batch_size = 10

def acc(target, pred):
    categorize = tf.cast(tf.greater(pred,0.5),tf.float32)
    same = tf.cast(tf.equal(categorize, target),tf.float32)
    return tf.reduce_mean(same)

if __name__ == "__main__":
    conv = conv_net()
    X1 = Input(shape=input_shape)
    X2 = Input(shape=input_shape)
    is_training = Input([1], dtype=bool)
    x1_tmp = Lambda(lambda x: transform_gate(x, 0.3, is_training[0][0]))(X1)
    x2_tmp = Lambda(lambda x: transform_gate(x, 0.3, is_training[0][0]))(X2)
    feature_x1 = conv(x1_tmp)
    feature_x2 = conv(x2_tmp)
    # Expect output < 0
    #pred = l2_distance(feature_pin, feature_pos) - l2_distance(feature_pin, feature_neg)
    mid_layer = Lambda(lambda x: tf.abs(feature_x1 - feature_x2))([feature_x1, feature_x2])
    pred = Dense(1,activation='sigmoid')(mid_layer)
    triplet_net = Model(input=[X1, X2, is_training],output=pred)
    optimizer = Adam(0.0001)
    triplet_net.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics=[acc])
    train_loader = DataGenerator(batch_size=batch_size)
    val_loader = DataGenerator(data_type='val',batch_size=batch_size, num_batch=50)
    save_model = cb.ModelCheckpoint('model/omniglot2', monitor='val_loss',save_best_only=True)
    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)
    triplet_net.fit_generator(generator=train_loader, validation_data=val_loader,  epochs=40, use_multiprocessing=True, workers=4, callbacks=[save_model, reduce_lr])
    triplet_net.save('model/siamese.h5')


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
