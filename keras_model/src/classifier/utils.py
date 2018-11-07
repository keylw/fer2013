import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def import_fer2013_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open("dataset/fer2013.csv") as f:
        content = f.readlines()
 
    lines = np.array(content)
 
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)

    x_train, y_train, x_test, y_test = [], [], [], []
    
    for i in range(1,num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")
        
            val = img.split(" ")
            pixels = np.array(val, 'float32')
        
            emotion = tf.keras.utils.to_categorical(emotion, 7)
            
            if 'Training' in usage:
                x_train.append(pixels)
                y_train.append(emotion)
            elif 'PublicTest' in usage:
                x_test.append(pixels)
                y_test.append(emotion)
        except:
            print("", end="")


    #data transformation for train and test sets
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_test' : x_test,
        'y_test' : y_test
        }

#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions, image):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))


    plt.figure(1)
    plt.subplot(211)
    plt.gray()
    plt.imshow(image)
    plt.title('face')

    plt.subplot(212)
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()
#------------------------------