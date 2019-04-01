import os
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN


def get_clusters(data):
    return DBSCAN(eps=0.8, min_samples=10).fit(data)


def show_clusters(db):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    n_0 = list(labels).count(-1)
    print(n_0)
    print(len(list(labels)))
    ax = plt.subplot(111, projection='3d')
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 2], xy[:, 3], xy[:, 4], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 2], xy[:, 3], xy[:, 4], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def get_clusters_data():
    path_dir = os.listdir('./model')
    for all_dir in path_dir:
        child = os.path.join('./model', all_dir)
        data = np.load(child)
        db = get_clusters(data)
        Y = db.labels_
        print(len(set(list(Y))))
        print('0:', list(Y).count(-1))
        print('all:', len(list(Y)))
        Y[Y != -1] = 1
        Y[Y == -1] = 0
        np.save('./label/' + all_dir[:-4] + '_label.npy', db.labels_)


if __name__ == '__main__':
    get_clusters_data()