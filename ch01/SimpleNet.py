from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboard import make_tensorboard


np.random.seed(1671)    # for reproducibility

# 各パラメータの設定
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10    # number of outputs = number of digits
OPTIMIZER = SGD()   # SGD optimizer
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

# MNIST データのロード
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
print(X_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000, )
print(X_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000, )
'''

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784

X_train = X_train.reshape(X_train.shape[0], RESHAPED)
X_test = X_test.reshape(X_test.shape[0], RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# MNISTデータの正規化(normalization)
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

'''
print(Y_train.shape)    # (60000, 10)
print(Y_test.shape)     # (10000, 10)
'''

# 10 outputs
# final stage is softmax
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# model のコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

# callbacks の設定
callbacks = [make_tensorboard(set_dir_name='SimpleNet')]

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          callbacks=callbacks,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])
