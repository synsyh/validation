import json

import matplotlib.pyplot as plt
import numpy as np

plt.figure()


# with open('./data/json_data_300.json') as f:
#     datas = json.loads(f.read())
#     datas = datas['trajectories']
#
# for data in datas:
#     xs = np.asarray([p['x'] for p in data])
#     ys = np.asarray([p['y'] for p in data])
#     plt.plot(xs, ys)

def draw_representative_line():
    with open('./data/vaptcha_output.json') as f:
        clus_datas = json.loads(f.read())

    for cd in clus_datas:
        xs = np.asarray([p['x'] for p in cd])
        ys = np.asarray([p['y'] for p in cd])
        plt.plot(xs, ys)
    print('cluster num:', len(clus_datas))
    plt.show()


def draw_clusters():
    with open('./data/test_output_clusters.json') as f:
        clus_datas = json.loads(f.read())

    for cd in clus_datas:
        for points in cd:
            plt.plot((points['start']['x'], points['start']['y']), (points['end']['x'], points['end']['y']))
    print('cluster num:', len(clus_datas))
    plt.show()


def draw_trajectory():
    with open('./data/json_data_30k_3000.json') as f:
        clus_datas = json.loads(f.read())['trajectories']

    for i, cd in enumerate(clus_datas):

        if i > 1:
            break
        xs = np.asarray([p['x'] for p in cd])
        ys = np.asarray([p['y'] for p in cd])
        plt.plot(xs, ys)
    print('cluster num:', len(clus_datas))
    plt.show()


# draw_clusters()
draw_representative_line()
# draw_trajectory()
