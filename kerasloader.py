import numpy as np
import keras
import random
from python.dataloader import loader

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_type='train', batch_size=20, num_batch=500, dim=(105,105), n_channels=1,
                 n_classes=30):
        'Initialization'
        self.type = data_type
        if self.type == 'train':
            self.is_training = np.array([True for _ in range(batch_size)])
        else:
            self.is_training = np.array([False for _ in range(batch_size)])
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_batch = num_batch
        self.y_target = np.zeros(self.batch_size)
        self.build_data(self.type)
        self.on_epoch_end()

        #TODO!!!!
        self.hard_batch = np.zeros(batch_size, *dim, n_channels)

    def build_data(self, data_type):
        self.class_data = np.array(loader(data_type, 'python/images_background'))
        self.n_classes = len(self.class_data)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X_pin, X_pos, X_neg = self.__data_generation()

        # X_pos = self.sess.run(self.trans_img_output, feed_dict={self.trans_img_input:X_pos})
        # X_pin = self.sess.run(self.trans_img_output, feed_dict={self.trans_img_input:X_pin})
        # X_neg = self.sess.run(self.trans_img_output, feed_dict={self.trans_img_input:X_neg})

        return [X_pin, X_pos, X_neg, self.is_training], self.y_target

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_pin = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_pos = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_neg = np.empty((self.batch_size, *self.dim, self.n_channels))
        # print(pos, neg)
        # print(self.class_data[pos][0].shape)
        # Generate data
        for i in range(self.batch_size):
            pos, neg = random.sample(range(self.n_classes), 2)
            pos_idx, pin_idx = random.sample(range(20),2)
            neg_idx = random.choice(range(20))
            X_pin[i,], X_pos[i,] = self.class_data[pos][pin_idx], self.class_data[pos][pos_idx]
            X_neg[i,] = self.class_data[neg][neg_idx]
        return X_pin, X_pos, X_neg
        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
