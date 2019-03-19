import json

import matplotlib.pyplot as plt
import numpy as np

plt.figure()

with open('./data/json_data_300.json') as f:
    datas = json.loads(f.read())
    datas = datas['trajectories']

for data in datas:
    xs = np.asarray([p['x'] for p in data])
    ys = np.asarray([p['y'] for p in data])
    plt.plot(xs, ys)

with open('./data/vaptcha_output.json') as f:
    clus_datas = json.loads(f.read())

for cd in clus_datas:
    xs = np.asarray([p['x'] for p in cd])
    ys = np.asarray([p['y'] for p in cd])
    # plt.plot(xs, ys)
print('cluster num:', len(clus_datas))
plt.show()
