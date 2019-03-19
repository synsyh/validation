import json
import math

import numpy as np
import matplotlib.pyplot as plt
from data_trans import analysis_data
from load_mongodb import MongoData
import pandas as pd

mongo_data = MongoData()
vps = mongo_data.get_mongodb_batch_info(size=30000)


def draw_time():
    times = []
    for vp in vps:
        time = int(vp['DrawTimeString'])
        times.append(time)
    time_array = np.asarray(times)
    plt.hist(time_array)
    plt.show()


# def draw_velocity():
#     max_velocity = []
#     for vp in vps:
#         points = get_velocity(analysis_data(vp['VerifyPath']))
#         v_max = max(points, key=lambda x: x['v'])['v']
#         if v_max > 7:
#             continue
#         max_velocity.append(v_max)
#     plt.hist(max_velocity, bins=50)
#     plt.show()


v_mean_list = []
v_max_list = []
a_mean_list = []

v_warp_dis_list = []
a_warp_dis_list = []

jsn = {'trajectories': []}

for i, vp in enumerate(vps):
    points = analysis_data(vp['VerifyPath'])
    df = pd.DataFrame(points)

    tmp = df['x'].shift(1)
    df['dis_x'] = df['x'] - tmp

    tmp = df['y'].shift(1)
    df['dis_y'] = df['y'] - tmp

    df['dis'] = np.sqrt(df['dis_x'] ** 2 + df['dis_y'] ** 2)

    tmp = df['time'].shift(1)
    df['dis_t'] = df['time'] - tmp

    df['v'] = df['dis'] / df['dis_t']
    tmp = df['v'].shift(1)
    df['a'] = (df['v'] - tmp) / df['dis_t'] * 100

    tmp = (df['x'] - df['x'].shift(2)) ** 2 + (df['y'] - df['y'].shift(2)) ** 2
    df['angle'] = (df['dis'].shift(-1) ** 2 + df['dis'] ** 2 - tmp.shift(-1)) / (2 * df['dis'].shift(-1) * df['dis'])
    df['angle_dif'] = df['angle'] - df['angle'].shift(1)
    if df['v'].max() > 4 or df['a'].max() > 4:
        continue
    qu = df['angle_dif'].quantile(.75)
    ql = df['angle_dif'].quantile(.25)
    qr = qu - ql
    u_range = qu + 2 * qr
    l_range = ql - 2 * qr
    warp_points = pd.concat([df[df.angle_dif > u_range], df[df.angle_dif < l_range]], axis=0).sort_values(by=['time'])
    warp_index = list(warp_points.index)
    warp_ranges = []
    warp_range = []
    k = 0
    while k < len(warp_index) - 1:
        if warp_index[k + 1] - warp_index[k] == 1:
            warp_range.append(warp_index[k])
        elif warp_index[k + 1] - warp_index[k] == 2:
            warp_range.append(warp_index[k])
            warp_range.append(warp_index[k] + 1)
        elif warp_index[k] - warp_index[k - 1] == 1 or warp_index[k] - warp_index[k - 1] == 2:
            warp_range.append(warp_index[k])
        else:
            if warp_range:
                warp_ranges.append(warp_range)
                warp_range = []
        if warp_index[k + 1] - warp_index[k] > 2:
            if warp_range:
                warp_ranges.append(warp_range)
                warp_range = []
        k += 1
    if len(warp_index) > 2:
        if warp_index[-1] - warp_index[-2] == 1 or warp_index[-1] - warp_index[-2] == 2:
            warp_range.append(warp_index[-1])
            warp_ranges.append(warp_range)

    v_mean = df['v'].mean()
    a_mean = df['a'].mean()
    a_list_tmp = []
    ax = plt.subplot(111)
    jsn_tmp = []
    for wr in warp_ranges:
        warp_df = df.loc[wr]
        # v_warp_mean = warp_df['v'].mean()
        # a_warp_mean = warp_df['a'].mean()
        # v_warp_dis = v_warp_mean - v_mean
        # a_warp_dis = a_warp_mean - a_mean
        # ax.plot(warp_df.angle_dif, warp_df.v)
        for s in warp_df.iterrows():
            jsn_tmp.append({'x': s[-1].angle_dif, 'y': s[-1].v})
        jsn['trajectories'].append(jsn_tmp)

    if i % 3000 == 0:
        with open('./traclus/data/json_data_30k_' + str(i) + '.json', 'w') as f:
            json.dump(jsn, f)
            jsn = {'trajectories': []}
        print(i)
    # v_mean_list.append(df['v'].mean())
    # v_max_list.append(df['v'].max())
    # a_mean_list.append(df['a'].mean())
# ax = plt.subplot(111, projection='3d')
with open('./traclus/data/json_data_30k_30000.json', 'w') as f:
    json.dump(jsn, f)

# 绘制散点图
# plt.show()
# plt.savefig('fig.png', bbox_inches='tight')

# X = []
# for i in range(len(v_mean_list)):
#     X.append(np.asarray([v_mean_list[i], v_max_list[i], a_mean_list[i]]))
# X = np.asarray(X)
# np.save('2000_3.npy', X)
