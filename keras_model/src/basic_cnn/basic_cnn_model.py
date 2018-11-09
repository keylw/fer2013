from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import optimizers
from keras import callbacks 

import numpy as np 
from dataset_api import fer2013
from architecture import demo_CNN_model
import utils

import sys
# sys.path.remove('/home/key/catkin_ws/devel/lib/python2.7/dist-packages')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 


# Variables
DATASET_PATH = "../../../dataset/fer2013.csv"
IMG_SIZE = 48

def train_model(save_mode=False, model_name="NaN"):
    num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
    batch_size = 64
    epochs = 1000

    dataset = fer2013(DATASET_PATH, val_ratio=0.05)
    dataset.reshape_size(IMG_SIZE)

    X_train = dataset.X_train
    Y_train = dataset.Y_train

    X_val = dataset.X_val
    Y_val = dataset.Y_val

    X_test = dataset.X_test
    Y_test = dataset.Y_test

    model = demo_CNN_model(input_size=(IMG_SIZE, IMG_SIZE, 1))

    # generators are used to handle large quanteties of data, important form machine learning
    train_generator = ImageDataGenerator().flow(X_train, Y_train, batch_size=batch_size)
    validation_generator = ImageDataGenerator().flow(X_val, Y_val, batch_size=batch_size)


    early_stopping_monitor = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    checkpointer = callbacks.ModelCheckpoint(filepath = "saved_models/model_weight.h5", verbose=1, save_best_only=True)

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        callbacks=[early_stopping_monitor, checkpointer],
                        steps_per_epoch=batch_size,
                        epochs=epochs)

    # ------------- evaluate the model--------------
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', 100*train_score[1])
    
    test_score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100*test_score[1])

    # -------------------save model------------------
    if save_mode:
        model_json = model.to_json()
        with open("saved_models/model_graph.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights("saved_models/model_weight.h5")
        print("Saved model to disk")

    return model

def load_model():
    json_file = open('saved_models/model_graph.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/model_weight.h5")
    print("Loaded model from disk")
    return loaded_model


def predict_emotion(test_image):
    model = load_model()

    test_image = cv2.resize(test_image, (IMG_SIZE, IMG_SIZE))
    test_image = (test_image[...,::-1].astype(np.float32)) 

    x = image.img_to_array(test_image)
    x = np.expand_dims(x, axis = 0)
    x /= 255

    custom = model.predict(x)

    print(custom[0])
    utils.plot_emotion_analysis(custom[0],test_image)

def example():    
    
    test_image = cv2.imread('images/krar.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    predict_emotion(test_image)

    # dataset = fer2013(DATASET_PATH, val_ratio=0.05)
    # dataset.reshape_size(IMG_SIZE)

    # predict_emotion(dataset.X_test[1954] * 255)
    # predict_emotion(dataset.X_test[1124] * 255)
    # predict_emotion(dataset.X_test[3434] * 255)


def main():
    train_model(save_mode=True, model_name="test")
    example()

if __name__ == '__main__':
    main()
