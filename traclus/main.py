import json
import time

import numpy as np
import pandas as pd

from data_trans import analysis_data
from load_mongodb import MongoData
from traclus.distance_trajectory import all_distance
from traclus.line_segment import Point, LineSegment
from traclus.line_segment_compute import get_trajectory_line_segments_from_points_iterable


def read_data(vp):
    jsn = {'trajectories': []}
    points = analysis_data(vp)
    df = pd.DataFrame(points)
    tmp = df['x'].shift(1)
    df['dis_x'] = df['x'] - tmp
    tmp = df['y'].shift(1)
    df['dis_y'] = df['y'] - tmp
    df['dis'] = np.sqrt(df['dis_x'] ** 2 + df['dis_y'] ** 2)
    tmp = df['time'].shift(1)
    df['dis_t'] = df['time'] - tmp
    df['v'] = df['dis'] / df['dis_t']
    tmp = (df['x'] - df['x'].shift(2)) ** 2 + (df['y'] - df['y'].shift(2)) ** 2
    df['angle'] = (df['dis'].shift(-1) ** 2 + df['dis'] ** 2 - tmp.shift(-1)) / (
            2 * df['dis'].shift(-1) * df['dis'])
    df['angle_dif'] = df['angle'] - df['angle'].shift(1)
    if df['v'].max() > 4:
        print('Eroor for velocity:%.2f' % df.v.max())
    qu = df['angle_dif'].quantile(.75)
    ql = df['angle_dif'].quantile(.25)
    qr = qu - ql
    u_range = qu + 1.5 * qr
    l_range = ql - 1.5 * qr
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
    if len(warp_index) > 2 and (warp_index[-1] - warp_index[-2] == 1 or warp_index[-1] - warp_index[-2] == 2):
        warp_range.append(warp_index[-1])
        warp_ranges.append(warp_range)
    jsn_tmp = []
    for wr in warp_ranges:
        warp_df = df.loc[wr]
        for s in warp_df.iterrows():
            jsn_tmp.append({'x': s[-1].angle_dif, 'y': s[-1].v})
        jsn['trajectories'].append(jsn_tmp)
        jsn_tmp = []
    return jsn


def read_model():
    model_path = '/Users/sunyuning/PycharmProjects/validation/traclus/data/vaptcha_clusters_3000.json'
    with open(model_path) as f:
        cluster_datas = json.loads(f.read())
    all_segments = []
    for points in cluster_datas:
        segments = []
        for point in points:
            ps = Point(point['start']['x'], point['start']['y'])
            pe = Point(point['end']['x'], point['end']['y'])
            line_segment = LineSegment(ps, pe)
            segments.append(line_segment)
        all_segments.append(segments)
    return all_segments


def read_cluster():
    model_path = './data/vaptcha_output.json'
    with open(model_path) as f:
        cluster_datas = json.loads(f.read())
    all_segments = []
    for cd in cluster_datas:
        segments = []
        for point in cd:
            ps = Point(point['start']['x'], point['start']['y'])
            pe = Point(point['end']['x'], point['end']['y'])
            line_segment = LineSegment(ps, pe)
            segments.append(line_segment)
        all_segments.append(segments)
    return all_segments


def check_trajectory(vp):
    data = read_data(vp)
    trajs = [[Point(j['x'], j['y']) for j in i] for i in data['trajectories']]
    model = read_model()
    n = p = 0.0
    for traj in trajs:
        tmp = get_trajectory_line_segments_from_points_iterable(traj)
        for k in range(len(tmp.line_segment)):
            n += 1
            t = tmp.line_segment[k]
            for i in range(len(model)):
                for j in range(len(model[i])):
                    d = all_distance(model[i][j], t)
                    if d < 0.02:
                        p += 1
                        break
                else:
                    continue
                break
    try:
        rate = p / n
    except ZeroDivisionError as e:
        print('No warp point')
        rate = 0
    return rate


if __name__ == '__main__':
    mongo_data = MongoData()
    vps = mongo_data.get_mongodb_batch_info(size=1)
    for i, vp in enumerate(vps):
        print(check_trajectory(vp['VerifyPath']))

