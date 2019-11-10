# 模型测试
import os
import logging
from keras.preprocessing import image


from utils import str_util, cv_util
import img_noise_generator

logging.basicConfig(level=logging.INFO,
                    filename='./logs/app.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def eval_noise_est(model, img_dir:str, height:int, width:int, max_sigma=1.0):
    logging.info('start evaluating...')
    for file in os.listdir(img_dir):
        if not str_util.is_img(file):
            continue
        img = cv_util.cv2_imread(os.path.join(img_dir, file), 1)
        x = image.img_to_array(img)
        lena = x.reshape(1, width, height, 3).astype('float32')
        lena = lena / 255
        # pdt = model.predict(np.array([img]))
        pdt = model.predict(lena)
        pdt = pdt[0]
        real_noise_sigma = img_noise_generator.parse_label(file)
        logging.info('file:{}, real_sigma:{}, predict:{}'.format(file, real_noise_sigma, pdt[0] * max_sigma))


if __name__ == '__main__':
    pass