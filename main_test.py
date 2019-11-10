import sys
import os



from img_noise_generator import img_generator as noise_img_gen
import model_helper

if __name__ == '__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    print('rootPath:{}'.format(rootPath))
    sys.path.append(rootPath)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.callbacks import ModelCheckpoint,TensorBoard
import gpu_monitor
from utils import str_util, config_util, logging_helper
from keras.models import Model
import my_callbacks
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import SGD


def binary_crossentropy(y_true, y_pred):
    # e = 1.3
    e = 1.0
    return K.mean(-(y_true * K.log(y_pred + K.epsilon()) +
                    e * (1 - y_true) * K.log(1 - y_pred + K.epsilon())),
                  axis=-1)
def bin_crt(e):
    def binary_crossentropy(y_true, y_pred):
        return K.mean(-(y_true * K.log(y_pred + K.epsilon()) +
                        e * (1 - y_true) * K.log(1 - y_pred + K.epsilon())),
                      axis=-1)
    return binary_crossentropy

if __name__ == '__main__':
    valid_gpus = gpu_monitor.get_valid_gpus(0.1)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_monitor.get_valid_gpus(0.1)[0])
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_cls_model3(img_width, img_height, channel_num):
    input_shape = (img_width, img_height, channel_num)
    input = Input(input_shape)
    x = Conv2D(64, (3, 3), name='cls-conv-0_0')(input)
    x = Conv2D(64, (3, 3), name='cls-conv-0_1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='cls-max_pool-0_0')(x)



    x = Conv2D(64, (3, 3), name='cls-conv-1_0')(x)
    x = Conv2D(64, (3, 3), name='cls-conv-1_1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='cls-max_pool-1_0')(x)

    x = Conv2D(128, (3, 3), name='cls-conv-2_0')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='cls-max_pool-2_0')(x)

    x = Flatten()(x)
    x = Dense(64, name='cls-dense-0')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='cls-dense-1')(x)
    x = Activation('sigmoid')(x)
    model = Model(input, x)
    model.compile(
        loss=binary_crossentropy,  # 多分类
        # loss='binary_crossentropy', #多分类
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model




def create_cls_model_newdata(img_width, img_height, channel_num=3):
    inputs = Input((img_width, img_height, channel_num), name='cls_input')
    # x = Conv2D(16, (3, 3), padding='same', name='share_conv1_1')(inputs)
    # x = BatchNormalization(name='share_bn_1')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, (3, 3), padding='same', name='share_conv1_2')(x)
    # x = BatchNormalization(name='share_bn_2')(x)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_1')(x)
    # x = BatchNormalization(name='cls_bn_3')(x)
    # x = Conv2D(32, (3, 3), padding='same', name='cls_conv_3')(inputs)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(32, (3, 3), padding='same', name='cls_conv_4')(x)
    # x = Activation('relu')(x)



    x = Conv2D(32, (3, 3), padding='same', name='share_conv1_1')(inputs)
    x = BatchNormalization(name='share_bn_1')(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), padding='same', name='share_conv1_2')(x)
    x = BatchNormalization(name='share_bn_2')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_2')(x)
    x = BatchNormalization(name='cls_bn_4')(x)
    x = Conv2D(64, (3, 3), padding='same', name='cls_conv_5')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='cls_conv_6')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='cls_conv_7')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_3')(x)
    x = BatchNormalization(name='cls_bn_5')(x)
    x = Conv2D(128, (3, 3), padding='same', name='cls_conv_8')(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='cls_conv_9')(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='cls_conv_10')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), name='cls_max_pool_4')(x)
    x = Conv2D(256, (3, 3), padding='same', name='cls_conv_11')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='cls_conv_12')(x)
    x = Activation('relu')(x)

    x = Flatten(name='cls_flatten')(x)
    x = Dense(64, name='cls_dense_1')(x)
    x = Activation('relu')(x)

    x = Dropout(0.5, name='cls_dropout')(x)
    x = Dense(1, name='cls_dense_2')(x)
    x = Activation('sigmoid', name='cls_out')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.summary()
    # x = Conv2D(64, (3, 3), padding='same', name='cls_1')(inputs)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(64, (3, 3), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    #
    # x = Conv2D(128, (3, 3), padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(128, (3, 3), padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(128, (3, 3), padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(256, (3, 3), padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(256, (3, 3), padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1, activation='sigmoid')(x)
    # model = Model(inputs=inputs, outputs=x)
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    #               metrics=['accuracy'])
    # model.summary()

    return model



def create_cls_model(img_width, img_height, channel_num):
    input_shape = (img_width, img_height, channel_num)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, name='conv0'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1)) #6分类
    model.add(Activation('sigmoid')) #采用Softmax

    model.compile(
        loss=binary_crossentropy,  # 多分类
        # loss='binary_crossentropy', #多分类
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model

def train(model, epochs, batch_size, img_height, img_width, channel_num, train_data_dir, validation_data_dir, callbacks=list()):
    train_list = os.listdir(os.path.join(train_data_dir, 'good'))
    test_list = os.listdir(os.path.join(validation_data_dir, 'good'))
    nb_train_samples = 0
    nb_validation_samples = 0
    for file in train_list:
        if str_util.is_img(file):
            nb_train_samples += 1
    for file in test_list:
        if str_util.is_img(file):
            nb_validation_samples += 1


    train_list = os.listdir(os.path.join(train_data_dir, 'bad'))
    test_list = os.listdir(os.path.join(validation_data_dir, 'bad'))
    for file in train_list:
        if str_util.is_img(file):
            nb_train_samples += 1
    for file in test_list:
        if str_util.is_img(file):
            nb_validation_samples += 1

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=True,  # apply ZCA whitenin
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        classes=['bad', 'good'],
        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        classes=['bad', 'good'],
        class_mode='binary')

    # validation_generator = test_datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     classes=['bad', 'good'],
    #     # class_mode='categorical')
    #     class_mode='binary')
    tensorboad = TensorBoard(log_dir='./test3_newdata_logs')
    checkpoint = ModelCheckpoint(filepath='test3_val_acc_max_2.h5',monitor='val_acc',mode='max' ,save_best_only='True')
    callback_lists=[tensorboad,checkpoint]
    plt_callback = my_callbacks.LossHistory()
    callback_lists.append(plt_callback)
    for cb in callbacks:
        callback_lists.append(cb)
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
        ,callbacks=callback_lists
    )
    # plt_callback.loss_plot()
    return model

def plot():
    model = create_cls_model2(243, 243, 3)
    plot_model(model=model, to_file='test3_model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


def create_noise_est_model(img_height, img_width, channel_num=2):
    input_shape = (img_height, img_width, channel_num)
    input = Input(input_shape)
    x = Conv2D(32, (3, 3))(input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(input, x)

    model.compile(
        loss='mse',  # 回归
        optimizer='sgd',
        metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # plot()
    # exit(0)
    params = config_util.get_param()
    batch_size = int(params.get('b', 3))
    epochs = int(params.get('e', 60))
    bias_e = float(params.get('bias', 1.3))
    logging_helper.debug('epochs={}, batchsize={}, bias_e={}'.format(epochs, batch_size, bias_e))
    img_width, img_height = 512, 512
    # train_data_dir = '../pre_data'
    # validation_data_dir = '../val_data'
    nb_train_samples = 220
    nb_validation_samples = 100
    #
    img_dir = './images/origin-gen'
    test_dir = './images/img_test'
    samples_num = 5200
    model = create_noise_est_model(img_height=img_height, img_width=img_width, channel_num=3)
    callbacks = [my_callbacks.NoiseEstCallback(img_dir=test_dir, height=img_height, width=img_width, max_sigma=60)]
    model.fit_generator(generator=noise_img_gen(img_dir=img_dir, batch_size=batch_size, sigma_max_value=60), steps_per_epoch=samples_num // batch_size, epochs=epochs, callbacks=callbacks)
    model_helper.save_model(model=model, struct_path='est_model.json', weight_path='est_model.h5')