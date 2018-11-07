from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import numpy as np 
from dataset_api import fer2013
from architecture import demo_CNN_model

DATASET_PATH = "../../../../dataset/fer2013.csv"

def train_model(save_mode=False, model_name="NaN"):
    num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
    batch_size = 128
    epochs = 50

    dataset = fer2013(DATASET_PATH, val_ratio=0.5)
    dataset.reshape_size(64)

    X_train = dataset.X_train
    Y_train = dataset.Y_train

    X_val = dataset.X_val
    Y_val = dataset.Y_val

    X_test = dataset.X_test
    Y_test = dataset.Y_test

    model = demo_CNN_model()

    # generators are used to handle large quanteties of data, important form machine learning
    train_generator = ImageDataGenerator().flow(X_train, Y_train, batch_size=batch_size)
    validation_generator = ImageDataGenerator().flow(X_val, Y_val, batch_size=batch_size)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
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
        with open("model_graph.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_weight.h5")
        print("Saved model to disk")

    return model


def load_model(model_path, weight_path):
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights/model.h5")
    print("Loaded model from disk")
    return loaded_model


def main():
    train_model(save_mode=True)

if __name__ == '__main__':
    main()