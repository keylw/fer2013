from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np
import utils 
import cv2

from face_detection.haar_cascade import Haar_cascade

# load json and create model
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights/model.h5")
print("Loaded model from disk")
 
# test_image = image.load_img("image/test5.jpg", grayscale=True, target_size=(48, 48))

test_image = cv2.imread('image/hk_1.jpg')
test_image, _ = Haar_cascade().get_face(test_image)
test_image = cv2.resize(test_image, (48, 48))
test_image = (test_image[...,::-1].astype(np.float32)) 

x = image.img_to_array(test_image)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = loaded_model.predict(x)
x = np.array(x, 'float32')
x = x.reshape([48, 48])

utils.emotion_analysis(custom[0],x)


