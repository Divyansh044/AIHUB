import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf # type: ignore
from tensorflow.keras import backend as K # type: ignore
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping   
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


train = pd.read_csv('ml_models/HWR/written_name_train_v2.csv')
valid = pd.read_csv('ml_models/HWR/written_name_validation_v2.csv')

train.dropna(axis=0, inplace=True)
valid.dropna(axis=0, inplace=True)

unreadable = train[train['IDENTITY'] == 'UNREADABLE']
unreadable.reset_index(inplace = True, drop=True)

train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']

train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()

train.reset_index(inplace = True, drop=True) 
valid.reset_index(inplace = True, drop=True)

def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

train_size = 50000
valid_size= 5000


train_x = []

for i in range(train_size):
    img_dir = 'ml_models/HWR/train/'+train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    train_x.append(image)

valid_x = []

for i in range(valid_size):
    img_dir = 'ml_models/HWR/validation/'+valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    valid_x.append(image)


train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
  
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
num_of_characters = len(alphabets) + 1  # +1 for CTC pseudo blank
num_of_timestamps = 64  
def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
    return np.array(label_num, dtype=np.int32)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1: 
            break
        else:
            ret += alphabets[ch]
    return ret

max_len = max(train['IDENTITY'].apply(len))

train_y = np.zeros((train_size, max_len), dtype=np.int32) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    train_label_len[i] = len(train.loc[i, 'IDENTITY'])
    train_y[i, 0:len(train.loc[i, 'IDENTITY'])]= label_to_num(train.loc[i, 'IDENTITY'])    

valid_y = np.ones([valid_size, max_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])    

# Model architecture

input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)



# Model creation
model = Model(inputs=input_data, outputs=y_pred)




def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='gtruth_labels', shape=[max_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam (learning_rate = 0.0001))


early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model_final.fit(
    x=[train_x, train_y, train_input_len, train_label_len], 
    y=train_output, 
    validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
    epochs=60, 
    batch_size=500,
    callbacks=[early_stopping]
)

model.save('handwriting_recognition_model.h5')































# Update label_to_num function to use -1 for characters not in alphabet
def label_to_num(label):
    label_num = []
    for ch in label:
        if ch in alphabets:
            label_num.append(alphabets.find(ch))
        else:
            label_num.append(-1)  # Use -1 for characters not in alphabet
    return np.array(label_num, dtype=np.int32)

# Update train_y generation to filter out negative labels
train_y = np.zeros((train_size, max_len), dtype=np.int32)
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    label_num = label_to_num(train.loc[i, 'IDENTITY'])
    non_negative_labels = label_num[label_num != -1]  # Filter out negative labels
    train_label_len[i] = len(non_negative_labels)
    train_y[i, :len(non_negative_labels)] = non_negative_labels

# Update valid_y generation to filter out negative labels
valid_y = np.zeros((valid_size, max_len), dtype=np.int32)
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    label_num = label_to_num(valid.loc[i, 'IDENTITY'])
    non_negative_labels = label_num[label_num != -1]  # Filter out negative labels
    valid_label_len[i] = len(non_negative_labels)
    valid_y[i, :len(non_negative_labels)] = non_negative_labels
