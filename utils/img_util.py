import os
import sys
if __name__ == '__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    print('rootPath:{}'.format(rootPath))
    sys.path.append(rootPath)

from utils import str_util
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from PIL import Image
import cv2

def color_kinds(img):
    color_set = set()
    if np.ndim(img) == 2:
        channel_num = 1
    elif np.ndim(img) >= 3:
        channel_num = np.size(img, np.ndim(img) - 1)
    else:
        print('unknow img format')
        return color_set

    print('img real channel:{}'.format(channel_num))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            color_str = ''
            if channel_num == 1:
                color_str = str(img[i][j])
            else:
                for c in range(channel_num):
                    color_str = color_str + ' ' + str(img[i][j][c])
            color_set.add('({})'.format(color_str))
    return color_set



def show_images(images, cols = 1, titles = None, save2dir=None, display=False, file_name='temp.png'):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None )or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1 ,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images /float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    if (not (save2dir == None) and os.path.exists(save2dir)):
        plt.savefig(os.path.join(save2dir, file_name))
        print('save plot to {}'.format(os.path.join(save2dir, file_name)))
    if display:
        plt.show()
    plt.close()

def rgb2gray(img, grey_level=256):
    if np.ndim(img) == 3:
        img = color.rgb2gray(img) * (grey_level - 1)
        img = np.uint8(img)
    return img

def convert_one_hot_2_class(label_path:str, target_path:str):
    # black: 0
    # red:1
    #
    img = Image.open(label_path)
    color_kind = set()
    print (img.size)
    print (img.getpixel((4,4)))
    width = img.size[0]#长度
    height = img.size[1]#宽度
    for i in range(0,width):#遍历所有长度的点
        for j in range(0,height):#遍历所有宽度的点
            data = (img.getpixel((i,j)))#打印该图片的所有点
            # print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            # print (data[0])#打印RGBA的r值
            color_kind.add(data)
            if(not (data == 1)):
                img.putpixel((i, j), 0)
            # if (data[0]>=170 and data[1]>=170 and data[2]>=170):#RGBA的r值大于170，并且g值大于170,并且b值大于170
            # if (not (data[0] == 255)):#RGBA的r值大于170，并且g值大于170,并且b值大于170
            #     img.putpixel((i,j),(0,0,0,0))#则这些像素点的颜色改成大红色
    # img = img.convert("RGB")#把图片强制转成RGB
    img.save(target_path)#保存修改像素点后的图片
    print(color_kind)

def one_hot_test():
    convert_one_hot_2_class('../BUS/GT_tumor2/case25.png', 'onehot27.png')

def batch_test():
    label_dir = '../BUS-255/GT_tumor2/case8.png'
    # label_dir = '../BUS/GT_tumor2'
    print(color_kinds(cv2.imread(label_dir, -1)))

    lb = Image.open(label_dir)
    lb = np.array(lb)
    print(color_kinds(lb))
if __name__ == '__main__':
    batch_test()