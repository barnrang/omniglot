# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="qRngFGVIBc32hxbR60UCEvavQ", project_name="omniglot")

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
from kerasloader import DataGenerator
from model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from transform import transform_gate

input_shape = (105,105,1)
batch_size = 20

if __name__ == "__main__":
    conv = conv_net()
    pin = Input(input_shape)
    pos = Input(input_shape)
    neg = Input(input_shape)
    is_training = Input([1], dtype=bool)
    pin_tmp = Lambda(lambda x: transform_gate(x, 0.5, is_training[0][0]))(pin)
    pos_tmp = Lambda(lambda x: transform_gate(x, 0.5, is_training[0][0]))(pos)
    neg_tmp = Lambda(lambda x: transform_gate(x, 0.5, is_training[0][0]))(neg)
    feature_pin = conv(pin_tmp)
    feature_pos = conv(pos_tmp)
    feature_neg = conv(neg_tmp)
    # Expect output < 0
    #pred = l2_distance(feature_pin, feature_pos) - l2_distance(feature_pin, feature_neg)
    pred = Lambda(lambda x: l1_distance(x[0], x[1]) - l1_distance(x[0], x[2]))([feature_pin, feature_pos, feature_neg])
    triplet_net = Model(input=[pin,pos,neg, is_training],output=pred)
    optimizer = Adam(0.00006)
    triplet_net.compile(loss = hinge_loss, optimizer=optimizer, metrics=[acc])
    train_loader = DataGenerator(batch_size=batch_size)
    val_loader = DataGenerator(data_type='val',batch_size=batch_size, num_batch=50)
    save_model = cb.ModelCheckpoint('model/omniglot2', monitor='val_loss',save_best_only=True)
    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)
    triplet_net.fit_generator(generator=train_loader, validation_data=val_loader,  epochs=40, use_multiprocessing=True, workers=4, callbacks=[save_model, reduce_lr])
    try:
        conv.save('model/conv/triplet.h5')
    except:
        print('save whole')
        triplet_net.save('model/conv/triplet.h5')


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
