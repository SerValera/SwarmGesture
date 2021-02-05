import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, GaussianDropout, Dropout,  Activation, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects


class_Dict = {'one':0, 'two':1, 'three':2, 'four':3, 'five':4, 'ok':5, 'rock':6, 'thumbs_up':7}
inv_class_Dict = {0:'one', 1:'two', 2:'three', 3:'four', 4:'five', 5:'ok', 6:'rock', 7:'thumbs_up'}

X = []
Y = []

dataset_DIR = 'database/'

files_dir = os.listdir(dataset_DIR)
files = os.listdir(dataset_DIR)

for i in range(len(files_dir)):
    files[i] = dataset_DIR + files_dir[i]

print(files)


for f in files:
  if f[-4:] == '.csv':
    data = pd.read_csv(f, header = None)
    for row in range(data.shape[0]):
      temp = []
      for pt in data.iloc[row]:
        #remove extra white spaces in the middle
        temp_pt = " ".join(pt.split())
        #remove brackets
        temp_pt = (temp_pt.strip(']')).strip('[')
        #remove trailing and leading white spaces
        temp_pt = temp_pt.strip()
        temp.append(float(temp_pt.split(' ')[0]))
        temp.append(float(temp_pt.split(' ')[1]))
      X.append(temp)
      Y.append(class_Dict[f[17:-4]])

## Augment data

print('Number of train samples before augmentation', len(X))

# Translation
for i in range(len(X)):
    for j in range(20):
        x_offset = np.random.randint(640) - 320
        y_offset = np.random.randint(480) - 240
        offset = [x_offset, y_offset] * 21
        X.append([a + b for a, b in zip(X[i], offset)])
        Y.append(Y[i])
        print(i, j)

# Create new samples with keypoint(joint) coordinate noise
for i in range(len(X)):
    for j in range(20):
        offset = [0] * 42
        index = np.random.randint(420)
        if index < 42:
            offset[index] = np.random.randint(20) - 10

        X.append([a + b for a, b in zip(X[i], offset)])
        Y.append(Y[i])
        print(i, j)

print('Number of train samples after augmentation', len(X))

X = np.array(X)
Y = np.array(Y)

X_train_max = X.max()
print('X_train_max: ', X_train_max)


# Normalization
# Only rough normalization, keypoint max could be bigger than image shape along x axis because of augmentation
X = (X / X.max()) - 0.5
# represent targets as one-hot-encoded
Y_onehot = to_categorical(Y, num_classes=8)



# Shuffle samples
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x_train, y_train = unison_shuffled_copies(X, Y_onehot)



# Best combination was relu/adam
activation = 'relu' # relu / tfa.activations.mish
drop_rate = 0.5

x1 = Input(shape=(42,))

x2 = Dense(168, activation=activation) (x1)
x2 = Dropout(drop_rate) (x2)

x3 = Dense(546, activation=activation) (x2)
x3 = Dropout(drop_rate) (x3)

x4 = Dense(8, activation='softmax') (x3)

model = Model(inputs=x1, outputs=x4)

radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=["acc"])

model.summary()



# Train
checkpoint = ModelCheckpoint('models/model2.h5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max')
history = model.fit(x_train, y_train, batch_size=256, epochs=14, validation_split=0.2, verbose=1, callbacks=[checkpoint])




# Save data
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()




#Create test data
X_test = []
Y_target = []

for f in files:
  if f[-4:] == '.csv':
    data = pd.read_csv(f, header = None)
    test = [100, 150, 200, 250]
    for row in test:
      temp = []
      for pt in data.iloc[row]:
        #remove extra white spaces in the middle
        temp_pt = " ".join(pt.split())
        #remove brackets
        temp_pt = (temp_pt.strip(']')).strip('[')
        #remove trailing and leading white spaces
        temp_pt = temp_pt.strip()
        temp.append(float(temp_pt.split(' ')[0]))
        temp.append(float(temp_pt.split(' ')[1]))
      X_test.append(temp)
      Y_target.append(f[17:-4])



X_test = np.array(X_test)
X_train_max = 983.09624295

# Normalization
# Only rough normalization, keypoint max could be bigger than image shape along x axis because of augmentation
X_test = (X_test / X_train_max) - 0.5

loaded_model = load_model('models/model2.h5')
Y_pred = loaded_model.predict(X_test)


# decode Y_pred
Y_pred_decoded = []
for i in range(Y_pred.shape[0]):
  Y_pred_decoded.append(inv_class_Dict[np.argmax(Y_pred[i])])

print('### Predicted ###')
print(Y_pred_decoded)
print('### Ground Truth ###')
print(Y_target)