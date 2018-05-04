import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
#import pandas as pd
BASE_PATH = "images_background"

def loader():
    "TODO!!!!"
    images = []
    labels = [] #keep in list of tuple
    folders_list = os.listdir(BASE_PATH)
    for folder in tqdm(folders_list):
        path1 = os.path.join(BASE_PATH, folder)
        for char_type in os.listdir(path1):
            path2 = os.path.join(path1, char_type)
            for image_name in os.listdir(path2):
                yield plt.imread(os.path.join(path2, image_name)).astype(np.int8), (folder, char_type)

if __name__ == "__main__":
    images, labels = zip(*list(loader()))
    index = -1
    print(images[index], labels[index])
    plt.imshow(images[index])
    plt.show()
