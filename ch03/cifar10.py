import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint

def network(input_shape, num_classes):
    model = Sequential()
    
    # extract image features by convolution and max pooling layers
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model

class CIFAR10Dataset():
    def __init__(self):
        self.image_shape = (32, 32, 3)
        self.num_classes = 10
        
    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]
        
        return x_train, y_train, x_test, y_test
        
    def preprocess(self, data, label_data=False):
        if label_data:
            # convert class vectors to binary clas matrices
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255 # convert value to 0-1 scale
            shape = (data.shape[0],) + self.image_shape # add dataset length to top
            data = data.reshape(shape)
            
        return data
