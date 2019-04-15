import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

def lenet(input_shape, num_classes):
    model = Sequential()
    
    # extract image features by convolution and max pooling layers
    model.add(Conv2D(20, kernel_size=5, padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=5, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dens(500, activation='relu'))
    model.add(Dens(num_classes))
    model.add(Activation('softmax'))
    
    return model

# MNISTSデータの前処理クラス
class MNISTDataset():
    def __init__(self):
        self.image_shape = (28, 28, 1)  # image is 28x28x1 (grayscale)
        self.num_classes = 10
        
    def get_batch(self):
        # MNISTデータのロード
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # 入力データの正規化
        x_train, x_test = [self.preprocess(d), for d in [x_train, x_test]]
        
        # 正解ラベルのバイナリクラスマトリックス化
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]
        
        return x_train, y_train, x_test, y_test
        
    def preprocess(self, data, label_data=False):
        if label_data:
            # convert class vectors to binary class matrices
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255 # convert the value to 0-1 scale
            shape = (data.shape[0],) + self.image_shape # add dataset length to top
            data = data.reshape(shape)
            
        return data
        
        
