import json
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from data_trans import analysis_data, get_velocity
from load_mongodb import MongoData
import pandas as pd
import matplotlib.pyplot as plt


def draw_time(vps):
    times = []
    for vp in vps:
        time = int(vp['DrawTimeString'])
        times.append(time)
    time_array = np.asarray(times)
    plt.hist(time_array)
    plt.show()


def draw_velocity(vps):
    max_velocity = []
    for vp in vps:
        points = get_velocity(analysis_data(vp['VerifyPath']))
        v_max = max(points, key=lambda x: x['v'])['v']
        if v_max > 7:
            continue
        max_velocity.append(v_max)
    plt.hist(max_velocity, bins=50)
    plt.show()


def calculate_velocity(df):
    tmp = df['x'].shift(1)
    df['dis_x'] = df['x'] - tmp
    tmp = df['y'].shift(1)
    df['dis_y'] = df['y'] - tmp
    df['dis'] = np.sqrt(df['dis_x'] ** 2 + df['dis_y'] ** 2)
    tmp = df['time'].shift(1)
    df['dis_t'] = df['time'] - tmp
    df['v'] = df['dis'] / df['dis_t']
    tmp = df.v.shift(1)
    df['a'] = (df.v - tmp) / df.dis_t * 100
    # TODO: velocity error
    if df['v'].max() > 15.0:
        print('Error v:%.2f' % df.v.max())
        df = None
    #     return df
    # if df.a.max() > 40.0:
    #     print('Error a:%.2f' % df.a.max())
    #     df = None
    return df


def calculate_angle(df):
    if 'v' not in df._series.keys():
        df = calculate_velocity(df)
    tmp = (df['x'] - df['x'].shift(2)) ** 2 + (df['y'] - df['y'].shift(2)) ** 2
    df['angle'] = (df['dis'].shift(-1) ** 2 + df['dis'] ** 2 - tmp.shift(-1)) / (2 * df['dis'].shift(-1) * df['dis'])
    df['angle_dif'] = df['angle'] - df['angle'].shift(1)
    return df


def get_warp(df):
    if df:
        calculate_angle(df)
    qu = df['angle_dif'].quantile(.75)
    ql = df['angle_dif'].quantile(.25)
    qr = qu - ql
    u_range = qu + 2 * qr
    l_range = ql - 2 * qr
    warp_points = pd.concat([df[df.angle_dif > u_range], df[df.angle_dif < l_range]], axis=0).sort_values(
        by=['time'])
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
    return warp_ranges


def get_warp_json(df):
    jsn = {'trajectories': []}
    warp_ranges = get_warp(df)
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


def get_dataframe(vp):
    # points = analysis_data(vp['VerifyPath'])
    points = analysis_data(vp)
    df = pd.DataFrame(points)
    return df


def save_npy(vps):
    X = []
    for vp in vps:
        df = get_dataframe(vp)
        df = calculate_velocity(df)
        if df is None:
            continue
        X.append(np.asarray([df.v.mean(), df.v.max(), df.a.mean()]))
    X = np.asarray(X)
    np.save(str(len(vps)) + '_3.npy', X)


def calculate_all(vps, i):
    X = []
    for vp in vps:
        try:
            df = get_dataframe(vp)
            df = calculate_velocity(df)
            if df is None:
                continue
            df = calculate_angle(df)
            ql = df.angle_dif.quantile(0.75)
            tmp = df.angle_dif[df.angle_dif > ql]
            n = len(tmp)
            if n < 3:
                continue
            start = tmp[:1].index.tolist()[0]
            end = tmp[-1:].index.tolist()[0]
            if start < 6:
                start = 6
            if len(df) - end < 3:
                continue
            start_df = df[2:start]
            end_df = df[end:]
            X.append(np.asarray(
                [df.v.mean(), df.v.max(), df.a.mean(), start_df.v.mean(), start_df.a.mean(), end_df.v.mean(),
                 end_df.a.mean()]))
        except Exception as e:
            print(e)
            continue
    X = np.asarray(X)
    np.save('./model/test_data_' + str(len(vps)) + '_' + str(i) + '.npy', X)


# useless
def draw_pic():
    ax = plt.subplot(111, projection='3d')
    plt.show()
    plt.savefig('fig.png', bbox_inches='tight')


if __name__ == '__main__':
    mongo_data = MongoData()
    for i in range(2):
        vps = mongo_data.get_mongodb_batch_info(size=10000)
        calculate_all(vps, i)
