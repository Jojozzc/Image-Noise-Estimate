import os
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import backend as K




def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def save_model(model, struct_path, weight_path):
    model.save_weights(weight_path)
    with open(struct_path, 'w') as struct_file:
        struct_file.write(model.to_json())



def net_from_json(path):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model



def load_model(struct_path, weight_path):
    if os.path.exists(struct_path) == True:
        # create model from JSON
        print('Reading network...\n')
        model = net_from_json(struct_path)
    else:
        print("Network structure file missing!.")
        exit(1)
    try:
        model.load_weights(weight_path)
    except:
        print("You must train model and get weight before test.")
        exit(1)
    return model
