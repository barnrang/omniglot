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
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from python.dataloader import loader

def conv_model():
    # inp = Input(shape=(105, 105, 1))
    model = Xception(include_top=False, weights='imagenet', input_shape=(105,105,3))
    # x = Conv2D(kernel_size=(3,3), filters=32, padding='valid', activation='relu')(inp)
    # x = MaxPooling2D()(x)
    # x = Conv2D(kernel_size=(3,3), filters=128, activation='relu')(x)
    # x = MaxPooling2D()(x)
    # x = Conv2D(kernel_size=(5,5), strides=(2,2), filters=128, activation='relu')(x)
    # x = MaxPooling2D()(x)
    # y = Conv2D(kernel_size=(5,5), strides=(2,2), filters=128, activation='relu')(x)
    # model = Model(inputs=inp, outputs=y)
    return model

def class_model(inp):
    x = Flatten()(inp)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(output_num, activation='softmax')(x)
    return x

images, labels = zip(*list(loader('python/images_background')))
images = np.expand_dims(images, axis=-1)
images = np.repeat(images, repeats=3, axis=-1)
print(images.shape)
main_labels, sub_labels= [x[0] for x in labels], [x[1] for x in labels]
encoder = LabelBinarizer()
enc_main_labels = encoder.fit_transform(main_labels)
output_num = len(np.unique(main_labels))
bottleneck_model = conv_model()
bottleneck_model.trainable = False
inp = Input(shape=(105,105,3))
features = bottleneck_model(inp)
prediction = class_model(features)
full_model = Model(inputs=inp, outputs=prediction)
adam = Adam(1e-3)
full_model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
full_model.fit(x=images, y=enc_main_labels, batch_size=32, epochs=100, validation_split=0.2)
