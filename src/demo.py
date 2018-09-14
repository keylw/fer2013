from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np 
import matplotlib.pyplot as plt
from models import demo_CNN_model
import utils 


#------------------------------
#variables
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 15
#------------------------------

dataset = utils.import_fer2013_dataset()

X_train = dataset['x_train']
Y_train = dataset['y_train']

X_test = dataset['x_test']
Y_test = dataset['y_test']

model = demo_CNN_model()

gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=batch_size)
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)

# evaluate the model
train_score = model.evaluate(X_train, Y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])


#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
#------------------------------

monitor_testset_results = False

if monitor_testset_results == True:
	#make predictions for test set
	predictions = model.predict(X_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			#print(i) #predicted scores
			#print(y_test[index]) #actual scores
			
			testing_img = np.array(X_test[index], 'float32')
			testing_img = testing_img.reshape([48, 48])
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
		index = index + 1

#------------------------------
#make prediction for custom image out of test set

img = image.load_img("image/krar.jpg", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48])

plt.gray()
plt.imshow(x)
plt.show()
#------------------------------
