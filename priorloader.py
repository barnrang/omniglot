import numpy as np
from keras.utils import np_utils
import keras
import random
from python.dataloader import loader

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_type='train', dim=(28,28), n_channels=1,
                way=20, shot=1, num_batch=500):
        'Initialization'
        self.type = data_type
        # if self.type == 'train':
        #     self.is_training = np.array([True for _ in range(batch_size)])
        # else:
        #     self.is_training = np.array([False for _ in range(batch_size)])
        self.dim = dim
        #self.batch_size = batch_size
        self.n_channels = n_channels
        self.num_per_class = 20
        self.num_batch = num_batch
        #self.y_target = np.zeros(self.batch_size)
        self.build_data(self.type)
        self.on_epoch_end()
        self.way = way
        self.shot = shot
        #TODO!!!!
        #self.hard_batch = np.zeros(batch_size, *dim, n_channels)

    def build_data(self, data_type):
        self.class_data = np.array(loader(data_type, 'python/images_background'))
        self.n_classes = len(self.class_data)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X_sample, X_query, label = self.__data_generation()

        # X_pos = self.sess.run(self.trans_img_output, feed_dict={self.trans_img_input:X_pos})
        # X_pin = self.sess.run(self.trans_img_output, feed_dict={self.trans_img_input:X_pin})
        # X_neg = self.sess.run(self.trans_img_output, feed_dict={self.trans_img_input:X_neg})

        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_sample = np.empty((self.way * self.shot, *self.dim, self.n_channels))
        X_query = np.empty((self.way * self.shot, *self.dim, self.n_channels))
        chosen_class = random.sample(range(self.n_classes), self.way)
        label = np.empty(self.way * self.shot)
        # print(pos, neg)
        # print(self.class_data[pos][0].shape)
        # Generate data
        for i in range(self.way):
            sample_idx = random.sample(range(self.num_per_class), 2 * self.shot)
            sample_data = self.class_data[chosen_class[i]][sample_idx]/255.
            X_sample[i * self.shot:(i+1) * self.shot] = sample_data[:self.shot]
            X_query[i * self.shot:(i+1) * self.shot] = sample_data[self.shot:]
            label[i * self.shot:(i+1) * self.shot] = i
        return X_sample, X_query, np_utils.to_categorical(label)
        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
