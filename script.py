# Plot mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load (downloaded) the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()




#IMPORT PACKAGES & CLASSES
import sys
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#initialize variables
# set batch and epoch sizes
batch_size = 200
epochs = 10

# fix random seed for reproducibility
seed = 93
numpy.random.seed(seed)

# input image dimensions
img_rows, img_cols = 28, 28

print("Variables intialized")


#RESHAPE DATA
# reshape to [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')
input_shape = (1, img_rows, img_cols)

# normalize i/p from 0-255 to 0-1
X_train /= 255
X_test  /= 255

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one hot encode o/p
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

from keras import models
from keras.models import load_model

model = load_model('w1_data.hdf5')

# Final evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)


print('Test Loss:     %.2f%%' % (scores[0]))
print('Test Accuracy: {:.2%}'.format(scores[1]))
print('Test Error:    {:.2%}'.format(1-scores[1]))
img = sys.argv[1]
from keras.preprocessing import image
test_image = image.load_img(img, grayscale = True, target_size = (28, 28))
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
a = result[0]
a = a.tolist()
print(a.index(1))
input()
