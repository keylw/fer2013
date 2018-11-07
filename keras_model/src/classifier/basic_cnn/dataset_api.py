# USING OPNECV ON ros
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.remove('/home/key/catkin_ws/devel/lib/python2.7/dist-packages')
import cv2 as cv2

import numpy as np 
import tensorflow as tf
from tqdm import tqdm


class fer2013:
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []

    def __init__(self, path_to_cvs, val_ratio=0.1):
        self.load_dataset(path_to_cvs, val_ratio)

    def load_dataset(self,path_to_cvs,val_ratio):
        with open(path_to_cvs) as f:
            print("reading files...")
            content = f.readlines()
 
        lines = np.array(content)

        num_of_instances = lines.size
        print("number of instances: ",num_of_instances)

        print("Loading dataset...")        
        for i in tqdm(range(1, num_of_instances)):
            try:
                emotion, img, usage = lines[i].split(",")
            
                val = img.split(" ")
                pixels = np.array(val, 'float32')
            
                emotion = tf.keras.utils.to_categorical(emotion, 7)
                
                if 'Training' in usage:
                    self.X_train.append(pixels)
                    self.Y_train.append(emotion)
                elif 'PublicTest' in usage:
                    self.X_test.append(pixels)
                    self.Y_test.append(emotion)
            except:
                print("", end="")

        n_train = len(self.Y_train)
        train_splitt = int(n_train*(1-val_ratio))

        self.X_val = self.X_train[train_splitt:]
        self.X_train = self.X_train[:train_splitt]
        self.Y_val = self.Y_train[train_splitt:]
        self.Y_train = self.Y_train[:train_splitt]


        #data transformation for train and test sets
        self.X_train = np.array(self.X_train, 'float32')
        self.Y_train = np.array(self.Y_train, 'float32')
        self.X_val = np.array(self.X_val, 'float32')
        self.Y_val = np.array(self.Y_val, 'float32')
        self.X_test = np.array(self.X_test, 'float32')
        self.Y_test = np.array(self.Y_test, 'float32')

        self.X_train /= 255 #normalize inputs between [0, 1]
        self.X_val /= 255 #normalize inputs between [0, 1]
        self.X_test /= 255

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 48, 48, 1)
        self.X_train = self.X_train.astype('float32')
        self.X_val = self.X_val.reshape(self.X_val.shape[0], 48, 48, 1)
        self.X_val = self.X_val.astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 48, 48, 1)
        self.X_test = self.X_test.astype('float32')


    def reshape_size(self, new_size):
        X_tr = np.zeros((self.X_train.shape[0],new_size, new_size) )
        X_te = np.zeros((self.X_test.shape[0],new_size, new_size) )
        X_va = np.zeros((self.X_val.shape[0],new_size, new_size) )


        for i,x in tqdm(enumerate(self.X_train)): 
            X_tr[i] = cv2.resize(x,(new_size, new_size) )

        for i,x in tqdm(enumerate(self.X_val)): 
            X_va[i] = cv2.resize(x,(new_size, new_size) )

        for i,x in tqdm(enumerate(self.X_test)): 
            X_te[i] = cv2.resize(x,(new_size, new_size) )

        self.X_train = X_tr
        self.X_val = X_va
        self.X_test = X_te

        # blute force 
        self.X_train = self.X_train.reshape(self.X_train.shape[0], new_size, new_size,  1)
        self.X_train = self.X_train.astype('float32')
        self.X_val = self.X_val.reshape(self.X_val.shape[0], new_size, new_size,  1)
        self.X_val = self.X_val.astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], new_size, new_size,  1)
        self.X_test = self.X_test.astype('float32')


    def print_example(self):
        self.reshape_size(128)
        for i in range(10):
            img = self.X_train[i]
            # res = cv2.resize(img,(244,244))
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for i in range(10):
            img = self.X_val[i]
            # res = cv2.resize(img,(244,244))
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    fer = fer2013("../../../dataset/fer2013.csv")
    fer.print_example()

if __name__ == '__main__':
    main()