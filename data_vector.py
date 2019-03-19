import math

from data_trans import analysis_data, get_velocity, scale
from load_mongodb import MongoData
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_data():
    mongo_data = MongoData()
    vps = mongo_data.get_mongodb_batch(size=2)
    vp = vps[0]
    points = analysis_data(vp)
    points = scale(points, if_zero=True)
    points = get_velocity(points)
    return points


def draw_vector(vps):
    n = len(vps)
    x = np.array([vp[1] for vp in vps])
    y_a = np.array([vp[0] for vp in vps])
    y_v = np.array([vp[2] for vp in vps])
    plt.plot(x, y_a)
    plt.plot(x, y_v)
    plt.show()


def trans2vector(points):
    vector_points = []
    for i in range(len(points) - 1):
        angle = (points[i + 1]['y'] - points[i]['y']) / (points[i + 1]['x'] - points[i]['x'])
        vector_points.append([sigmoid(angle), sigmoid(points[i]['time'] / 1000), sigmoid(points[i]['v'])])
    return vector_points


def trans2vector_matrix(points, max_len, step=1):
    n = int((len(points) - max_len) / step - 1)
    x = np.zeros((n, max_len, 3))
    y = np.zeros((n, 3))
    vector_points = []
    for i in range(len(points) - 1):
        try:
            angle = (points[i + 1]['y'] - points[i]['y']) / math.sqrt(
                (points[i + 1]['x'] - points[i]['x']) ** 2 + (points[i + 1]['y'] - points[i]['y']) ** 2)
        except ZeroDivisionError:
            angle = 10
        vector_points.append([sigmoid(angle), sigmoid(points[i]['time'] / 1000), sigmoid(points[i]['v'])])
    j = 0
    for i in range(n):
        x[i] = vector_points[j:j + max_len]
        y[i] = vector_points[j + max_len]
        j += step
    return x, y


def get_regression_data(verify_path):
    points = get_velocity(scale(analysis_data(verify_path['VerifyPath'])))


if __name__ == '__main__':
    datas = get_data()
    vps = trans2vector(datas)
    draw_vector(vps)
