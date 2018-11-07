import numpy as np 
import cv2
import tensorflow as tf
from tqdm import tqdm

class fer2013:
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    def __init__(self, path_to_cvs):
        self.load_dataset(path_to_cvs)

    def load_dataset(self,path_to_cvs):
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

        #data transformation for train and test sets
        self.X_train = np.array(self.X_train, 'float32')
        self.Y_train = np.array(self.Y_train, 'float32')
        self.X_test = np.array(self.X_test, 'float32')
        self.Y_test = np.array(self.Y_test, 'float32')

        self.X_train /= 255 #normalize inputs between [0, 1]
        self.X_test /= 255

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 48, 48, 1)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 48, 48, 1)
        self.X_test = self.X_test.astype('float32')

        X_tr = np.zeros((self.X_train.shape[0],64,64))
        X_te = np.zeros((self.X_test.shape[0],64,64))

        for i,x in tqdm(enumerate(self.X_train)): 
            X_tr[i] = cv2.resize(x,(64,64))

        for i,x in tqdm(enumerate(self.X_test)): 
            X_te[i] = cv2.resize(x,(64,64))

        self.X_train = X_tr
        self.X_test = X_te

        # blute force 
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 64, 64, 1)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 64, 64, 1)
        self.X_test = self.X_test.astype('float32')

    def print_example(self):

        for i in range(30):
            img = self.X_train[i]
            # res = cv2.resize(img,(244,244))
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    fer = fer2013("fer2013.csv")
    fer.print_example()

if __name__ == '__main__':
    main()




