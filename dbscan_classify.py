import os

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_data():
    path_dir = os.listdir('./model')
    child = os.path.join('./model', path_dir[0])
    data = np.load(child)
    for k, all_dir in enumerate(path_dir):
        child = os.path.join('./model', all_dir)
        if k == 0:
            continue
        else:
            data = np.concatenate((data, np.load(child)), axis=0)
    path_dir = os.listdir('./label')
    child = os.path.join('./label', path_dir[0])
    label = np.load(child)
    for k, all_dir in enumerate(path_dir):
        child = os.path.join('./label', all_dir)
        if k != 0:
            label = np.concatenate((label, np.load(child)), axis=0)

    label_index = np.argwhere(label == 1)
    label_index = label_index[:int(len(label_index) / 2)]
    data = np.delete(data, label_index, axis=0)
    label = np.delete(label, label_index, axis=0)
    return data, label


# data = np.load('./100000_7.npy')
# label = np.load('./label_100000_7.npy')
data, label = get_data()
# print(list(y_train).count(1) / len(list(y_train)))
label[label == 2] = 1
label[label == 3] = 1
print(list(label).count(1)/len(list(label)))
model = Sequential()
model.add(Dense(21, activation='sigmoid', input_dim=7))
model.add(Dense(7, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(data, label,
          batch_size=32,
          epochs=15,
          verbose=2,
          validation_split=0.2)
model.save('./dbscan.h5')
# model = keras.models.load_model('./dbscan2.h5')
# test_data = np.load('./test_data_10000_0.npy')
# test_label = np.load('./test_label_10000_0.npy')
# loss_and_metrics = model.evaluate(test_data, test_label)
# print(loss_and_metrics)
