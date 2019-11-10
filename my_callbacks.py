import keras
import matplotlib.pyplot as plt
import os


import eval
from mymodel import evaluate, model_helper
import datetime
from mymodel import pair2_test_new_data
import traceback
class NoiseEstCallback(keras.callbacks.Callback):

    def __init__(self, img_dir:str, height, width, max_sigma, model_cache_dir='./model_cache'):
        super(NoiseEstCallback, self).__init__()
        self.model_cache_dir = model_cache_dir
        self.img_dir = img_dir
        self.height = height
        self.width = width
        self.max_sigma = max_sigma
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir)
    def on_train_begin(self, logs=None):
        print('start training')

    def on_epoch_end(self, epoch, logs=None):
        eval.eval_noise_est(model=self.model, img_dir=self.img_dir, max_sigma=self.max_sigma, height=self.height, width=self.width)


if __name__ == '__main__':
    pass
