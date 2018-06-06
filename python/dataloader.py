import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
#import pandas as pd
BASE_PATH = "images_background"
TRAIN_CLASS = 25

def loader(data_type='train',path=None):
    "TODO!!!!"
    index = 0
    images = []
    if path is None:
        path = BASE_PATH
    folders_list = os.listdir(path)
    folders_list.sort()
    if data_type == 'train':
        folders_list = folders_list[:TRAIN_CLASS]
    else:
        folders_list = folders_list[TRAIN_CLASS:]
    for folder in tqdm(folders_list):
        path1 = os.path.join(path, folder)
        try: #In case of invalid folder
            for char_type in os.listdir(path1):
                path2 = os.path.join(path1, char_type)
                for rot in [0,90,180,270]:
                    class_image = []
                    for image_name in os.listdir(path2):
                        image = plt.imread(os.path.join(path2, image_name)).astype(np.int8)
                        image = imresize(image,(28,28))
                        if data_type == 'train':
                            image = rotate(image, rot)
                        image = np.expand_dims(image, axis=-1)
                        class_image.append(image)
                images.append(class_image)
        except:
            continue
    return np.array(images)


if __name__ == "__main__":
    images = loader()
    index = -1
    print(images[0][index])
    plt.imshow(images[index])
    plt.show()
