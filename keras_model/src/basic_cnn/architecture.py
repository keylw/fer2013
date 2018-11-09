from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dropout
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)



def NN_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
        
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def demo_CNN_model(input_size):
    model = Sequential()
 
    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_size))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
    
    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(3, 3)))

    model.add(Flatten())
    
    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(7, activation='softmax'))
    return model