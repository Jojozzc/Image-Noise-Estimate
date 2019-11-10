from keras.utils.np_utils import to_categorical
import numpy as np
import cv2
from PIL import Image

def rgb2one_hot(img, rgb2class=None):
    if rgb2class == None:
        rgb2class = {'128,0,0': 0, '128,128,0': 1, '0,0,128': 2, '0,128,0': 3, '0,0,0': 4}
    # 128,0,0 red
    # 128,128,0 yellow
    # 0,0,128 blue
    #  0, 128, 0 green
    # 0 0 0 black
    img = np.array(img)
    label = img.flatten()
    label = [rgb2class['{},{},{}'.format(label[i], label[i + 1], label[i + 2])] for i in range(0, len(label), 3)]
    label = to_categorical(label, 5)
    return label


def resize_img(img, target_size, interpolation=cv2.INTER_AREA):
    img = cv2.resize(img, dsize=target_size, interpolation=interpolation)
    return img

def rgb2categorical(label_path, shape, one_hot_num):
    label = Image.open(label_path)
    label = label.resize(shape)
    label = np.array(label).flatten()
    return to_categorical(label, one_hot_num)