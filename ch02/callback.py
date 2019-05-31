from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
import os

BATCH_SIZE = 128
NUM_EPOCHS = 20
MODEL_DIR = "/tmp"

# MNIST データのロード
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

# MNIST データの整形
# ベクトル化と正規化
Xtrain = Xtrain.reshape(60000, 784).astype("float32") / 255
Xtest = Xtest.reshape(10000, 784).astype("float32") / 255

# カテゴリカルに変換
Ytrain = np_utils.to_categorical(ytrain, 10)
Ytest = np_utils.to_categorical(ytest, 10)

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# モデルの構築
model = Sequential()
model.add(Dense(512, input_shape=(784, ), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# モデルのコンパイル
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5")
filepath=".\\\\tmp\\\\model-{epoch:02d}.h5"

print(os.path)

# save best models
checkpoint = ModelCheckpoint(filepath=filepath)

# fit
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1, callbacks=[checkpoint])
