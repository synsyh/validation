import data_trans
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import load_mongodb
from data_trans import get_velocity

matplotlib.use('TkAgg')


def color_gradient(steps, gradient):
    color = [int(color_start[i] + steps * gradient[i]) for i in range(3)]
    color_hex = '#'
    for i in range(3):
        h = str(hex(color[i]))[2:]
        if len(h) == 1:
            h = '0' + h
        color_hex += h
    return color_hex


def draw(data):
    if type(raw_data).__name__ == 'str':
        datas = data_trans.analysis_data(raw_data)
    else:
        datas = raw_data
    datas = get_velocity(datas)

    for i, data in enumerate(datas):
        print(data['v'])
        plt.scatter(data['x'], data['y'], c=color_gradient(data['v']))
    plt.show()
    print()


if __name__ == '__main__':
    data = np.zeros((9, 128, 128, 3))
    verify_paths, object_id = load_mongodb.get_mongodb_batch(object_id='5b7faf5dcc494a07782b153d')
    print(object_id)
    for i, raw_data in enumerate(verify_paths):
        points = data_trans.analysis_data(raw_data)
        points = data_trans.scale(points)
        points = get_velocity(points)
        for point in points:
            data[i][point['x']][point['y']] = [1, point['time'] / 100, point['v']]

    color_start = (0, 0, 255)
    color_end = (255, 0, 0)
    all_gradient = [color_end[i] - color_start[i] for i in range(3)]
    r = 0.1
    plt.figure()
    for i in range(9):
        v_max = -1
        for x in range(128):
            for y in range(128):
                if data[i][x][y][-1] > v_max:
                    v_max = data[i][x][y][-1]
        gradient = [g / v_max for g in all_gradient]
        plt.subplot(3, 3, i + 1)
        plt.title('max_v:' + str(v_max)[:4])
        for x in range(128):
            for y in range(128):
                if data[i][x][y][0] == 0:
                    continue
                else:
                    steps = data[i][x][y][2]
                    c = color_gradient(steps, gradient)
                    plt.scatter(x, y, c=c)
    plt.show()
