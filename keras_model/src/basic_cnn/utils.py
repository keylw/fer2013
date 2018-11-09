import numpy as np
import matplotlib.pyplot as plt


#function for drawing bar chart for emotion preditions
def plot_emotion_analysis(emotions, image):
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