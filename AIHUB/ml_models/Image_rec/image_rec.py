import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers, optimizers # type: ignore
from tensorflow.keras import datasets, layers, models # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.metrics import accuracy_score
from PIL import ImageFont
import warnings 
warnings.filterwarnings('ignore')




from tensorflow.keras.datasets import cifar10 # type: ignore
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train/255
X_test = X_test/255

# One-Hot-Encoding
Y_train_en = to_categorical(Y_train,10)
Y_test_en = to_categorical(Y_test,10)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


model = Sequential()
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])    

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Fit the model with early stopping
history = model.fit(X_train, Y_train_en, epochs=10, verbose=1, 
                    validation_data=(X_test, Y_test_en), callbacks=[early_stopping])

model.save("img_classify.keras")