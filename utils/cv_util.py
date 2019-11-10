import numpy as np
import cv2
import os

def cv2_imread(filename, mode=-1):
    # 0:gray
    # 1:rgb
    # -1 any, read as the mode of source itself.
    cv_img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), mode)
    return cv_img

def cv2_imwrite(filename, img, suffix=None):
    if suffix is None:
        suffix = os.path.splitext(filename)[-1]
    if suffix == '':
        suffix = '.png'
    cv2.imencode(suffix, img)[1].tofile(filename)

def resize_contours(cnts, cnts_origin_shape_WxH:tuple, to_shape_WxH:tuple):
    w_resize_rate = to_shape_WxH[0] / cnts_origin_shape_WxH[0]
    h_resize_rate = to_shape_WxH[1] / cnts_origin_shape_WxH[1]
    rate = (w_resize_rate, h_resize_rate)
    for i in range(len(cnts)):
        cnt = cnts[i] * rate
        cnts[i] = np.array(cnt, dtype='int')
    return cnts

def ttttest():
    cnts_path = '../high_speed_train/newUITestData2/.DETECTION_RESULT/K696+203_S0_P001_T235240784_ZCA1_jpg/cnts.npy'
    cnts = np.load(cnts_path)
    print(cnts.dtype)
    # print(cnts)
    print(len(cnts))
    res = resize_contours(cnts, (6600, 3300), (800, 600))
    print(res)

def test_for_cv_img():
    # gray
    img_path = '照片.jpg'
    img = cv2_imread(img_path, -1)
    cv2_imwrite('test.png', img, '')
if __name__ == '__main__':
    test_for_cv_img()