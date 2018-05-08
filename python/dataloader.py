import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
#import pandas as pd
BASE_PATH = "images_background"

def loader(path=None):
    "TODO!!!!"
    images = []
    labels = [] #keep in list of tuple
    if path is None:
        path = BASE_PATH
    folders_list = os.listdir(path)
    for folder in tqdm(folders_list):
        path1 = os.path.join(path, folder)
        try: #In case of invalid folder
            for char_type in os.listdir(path1):
                path2 = os.path.join(path1, char_type)
                class_image = []
                for image_name in os.listdir(path2):
                    image = plt.imread(os.path.join(path2, image_name)).astype(np.int8)
                    image = np.expand_dims(image, axis=-1)
                    class_image.append(image)
                images.append(class_image)
        except:
            continue
    return images
    

if __name__ == "__main__":
    images, labels = zip(*list(loader()))
    index = -1
    print(images[index], labels[index])
    plt.imshow(images[index])
    plt.show()
