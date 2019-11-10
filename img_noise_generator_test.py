from img_noise_generator import img_generator

if __name__ == '__main__':
    img_dir = './origin'
    x_y = img_generator(img_dir=img_dir, batch_size=3, sigma_max_value=60)
    x, y = x_y.__next__()
    print('ok')