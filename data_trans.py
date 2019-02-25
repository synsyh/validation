# 解密数据
import math
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Sample = "abcdefgh234lmntuwxyz"
file_path = './data/track1000'


def get_velocity(ps):
    for i in range(len(ps) - 2):
        v = math.sqrt((ps[i + 2]['x'] - ps[i]['x']) ** 2 + (ps[i + 2]['y'] - ps[i]['y']) ** 2) / (
                ps[i + 2]['time'] - ps[i]['time'])
        ps[i + 1]['v'] = v
    ps[0]['v'] = 0
    ps[-1]['v'] = ps[-2]['v']
    return ps


def trans_from_data(code):
    scale = len(Sample)
    length = int(len(code))
    result = 0
    for i in range(int(length / 3)):
        if i == 0 and code[i] == '_':
            result = Sample.index(code[i + 1]) * \
                     scale + Sample.index(code[i + 2])
        else:
            result = Sample.index(code[i]) * scale * scale + Sample.index(
                code[i + 1]) * scale + Sample.index(code[i + 2])
    return result


def analysis_data(code):
    length = int(len(code) / 9)
    points = []
    for i in range(length):
        ix = i * 3
        iy = i * 3 + length * 3
        it = i * 3 + 2 * length * 3
        x = code[ix:ix + 3]
        y = code[iy:iy + 3]
        t = code[it:it + 3]
        points.append({'x': int(trans_from_data(x)), 'y': int(trans_from_data(y)), 'time': int(trans_from_data(t))})

    return points


def get_datas(file_path):
    with open(file_path, 'r') as f:
        for line in f.readline():
            points = analysis_data(line)
            points = sorted(points, key=lambda x: x['time'])
            max_y = max(points, key=lambda x: x['y'])['y']
            min_y = min(points, key=lambda x: x['y'])['y']
            for point in points:
                xs.append(point['x'])
                ys.append(-point['y'] + max_y + min_y)


def scale(points):
    boundary = 120
    x_max, x_min, y_max, y_min = -1, float('inf'), -1, float('inf')
    for point in points:
        if point['x'] < x_min:
            x_min = point['x']
        if point['x'] > x_max:
            x_max = point['x']
        if point['y'] < y_min:
            y_min = point['y']
        if point['y'] > y_max:
            y_max = point['y']
    if x_max - x_min > y_max - y_min:
        dis = x_max - x_min
    else:
        dis = y_max - y_min
    scale_ratio = boundary / dis
    new_points = []
    for point in points:
        new_points.append({'x': int((point['x'] - x_min) * scale_ratio), 'y': int((point['y'] - y_min) * scale_ratio),
                           'time': point['time']})
    return new_points


def trans2matrix(points):
    data = np.zeros((128, 128, 3))
    for point in points:
        data[int(point['x'])][int(point['y'])] = [1, point['time'] / 100,
                                                  point['v']]
    return data


if __name__ == '__main__':
    ps = analysis_data(
        'cczccnccgcbzcblcaycamcabbzubzhbzabytby2bycbxxbxlbxgbxbbwwbwtbwnbwnbwubwzbxebx4bxwbygbyubzdbz2bzmbzwbzzcaeca3ca3camcamcancaucawcaycbacbccbfcb2cb3_za_yw_yw_yu_yu_ym_yl_yg_ye_yb_xy_xu_xn_x4_x3_xh_xd_xa_wn_w2_wd_uu_u3_ud_ty_tu_tt_tt_tu_ub_uf_u3_ut_wa_wg_wt_xd_xn_xz_ye_ym_yx_zc_z2_zxbadba3bat_aa_fw_hw_3z_ln_td_ua_x4_zdbawbc4bedbfwb3dbm4bxlcclc2hc4xcmlcteczedfbdhldmlduxdzeeclehle4yemleteeuyexlezefaxfclfeeffyfhlf4yfmlftefuyfzfgaxgeehbm')
    ps = scale(ps)
    ps = sorted(ps, key=lambda x: x['time'])

    xs = []
    ys = []

    zx = []
    zy = []

    mx = []
    my = []

    maxy = max(ps, key=lambda x: x['y'])['y']
    miny = min(ps, key=lambda x: x['y'])['y']
    for p in ps:
        xs.append(p['x'])
        ys.append(-p['y'] + miny + maxy)
    zx.append(xs[0])
    zy.append(ys[0])

    lindex = xs.__len__() - 1

    mx.append(xs[lindex])
    my.append(ys[lindex])

    fig = plt.figure()
    plt.plot(xs, ys)
    plt.axis('off')
    plt.show()
