from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
from data_trans import analysis_data, scale, get_velocity
from data_vector import trans2vector_matrix, get_regression_data
from load_mongodb import MongoData

mongo_data = MongoData()
k = 1
n_batch = 1000
vps = mongo_data.get_mongodb_batch(size=n_batch, if_points=1)
for i in range(k):
    vps = mongo_data.get_mongodb_batch(size=n_batch)
    for vp in vps:
        raw_data = vp['VerifyPath']
        points = analysis_data(raw_data)
        if len(points) < 40:
            continue

        x_p, y_p = get_regression_data(vp)
        x = np.concatenate((x, x_p), axis=0)
        y = np.concatenate((y, y_p), axis=0)

print('points length:', len(x))
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, 3)))
model.add(Dropout(0.05))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Prediction after Epoch: %d' % epoch)
    vp = mongo_data.get_mongodb_batch(size=1)[0]
    points = get_velocity(scale(analysis_data(vp)))
    try:
        xs, ys = trans2vector_matrix(points, maxlen)
    except TypeError as e:
        vp = mongo_data.get_mongodb_batch(size=1)[0]
        points = get_velocity(scale(analysis_data(vp)))
        xs, ys = trans2vector_matrix(points, maxlen)
    plt.figure()
    x_axis = range(len(xs))
    y1 = []
    y11 = []
    y2 = []
    y22 = []
    y3 = []
    y33 = []
    for i, x in enumerate(xs):
        x = x.reshape(1, maxlen, 3)
        preds = model.predict(x, verbose=0)[0]
        y1.append(preds[0])
        y11.append(ys[i][0])
        y2.append(preds[1])
        y22.append(ys[i][1])
        y3.append(preds[2])
        y33.append(ys[i][2])
        print('prediction :%.4f' % preds[0] + '\t%.4f' % preds[1] + '\t%.4f' % preds[2])
        print('real       :%.4f' % ys[i][0] + '\t%.4f' % ys[i][1] + '\t%.4f' % ys[i][2])
    plt.subplot(131)
    plt.plot(x_axis, y1, label='prediction')
    plt.plot(x_axis, y11, label='real')
    plt.subplot(132)
    plt.plot(x_axis, y2, label='prediction')
    plt.plot(x_axis, y22, label='real')
    plt.subplot(133)
    plt.plot(x_axis, y3, label='prediction')
    plt.plot(x_axis, y33, label='real')
    plt.legend()
    plt.show()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

optimizer = RMSprop(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.fit(x, y,
          batch_size=128,
          epochs=60, verbose=1,
          callbacks=[print_callback])
