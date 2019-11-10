import numpy as np
import os
from utils import str_util, cv_util


# 生成图像和图像标签
def img_generator(img_dir, batch_size=32, sigma_max_value=60):
    files = os.listdir(img_dir)
    img_files = []
    for file in files:
        if os.path.isfile(os.path.join(img_dir, file)) and str_util.is_img(file):
            img_files.append(file)
    indexs = list(range(len(img_files)))
    while True:
        np.random.shuffle(indexs)
        for i in range(0, len(indexs), batch_size):
            data = list()
            label = list()
            for j in range(i, i + batch_size):
                idx = j
                if idx >= len(indexs):
                    idx = idx % len(indexs)
                img = cv_util.cv2_imread(os.path.join(img_dir, img_files[indexs[idx]]), 1)
                img = np.array(img).astype('float32')
                img = np.multiply(img, 1 / 255.0)
                data.append(np.array(img))
                label.append(parse_label(img_files[idx]) / sigma_max_value)
            data = np.array(data)
            label = np.array(label)


            yield data, label

# 文件的命名格式为噪声大小-文件名，例如20-lena.bmp, 25-dx.png
def parse_label(file_name : str):
    str_group = file_name.split('-')
    return float(str_group[0])