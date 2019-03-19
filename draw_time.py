# 实时再绘制轨迹
import data_trans
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def draw(raw_data):
    if type(raw_data).__name__ == 'str':
        data = data_trans.analysis_data(raw_data)
    else:
        data = raw_data
    xs = [data[0]['x']]
    ys = [-data[0]['y']]
    for i in range(len(data) - 1):
        dis_x = data[i + 1]['x'] - data[i]['x']
        dis_y = data[i + 1]['y'] - data[i]['y']
        n = (data[i + 1]['time'] - data[i]['time']) / 10
        step_x = dis_x / n
        step_y = dis_y / n
        tmp_x = data[i]['x']
        tmp_y = data[i]['y']
        for j in range(int(n)):
            tmp_x += step_x
            tmp_y += step_y
            xs.append(tmp_x)
            ys.append(-tmp_y)
    xs.append(data[-1]['x'])
    ys.append(-data[-1]['y'])
    max_x = max(xs)
    min_x = min(xs)
    max_y = max(ys)
    min_y = min(ys)
    plt.ion()
    for i in range(int(data[-1]['time'] / 10) + 1):
        plt.clf()
        plt.plot(max_x, max_y)
        plt.plot(min_x, min_y)
        plt.plot(xs[:i], ys[:i])
        plt.pause(0.001)
        plt.ioff()
    plt.clf()
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    raw_data = 'bc3bctbd4bdubecbehbewbfabfbbfbbfabeube4befbdybdnbddbctbchbccbbtbb3bbdbaxbambahbab_zx_zu_zm_z4_z4_z4_z2_zg_zg_yh_y4_yw_yx_za_zc_zg_z4_zubabbahbanbaxbbabbcbbgbbhbbhbbhbbgbbdbbbbazbaxbaubanbalbahbac_zw_zl_zf_yy_yn_y2_yc_aa_ea_3n_mg_u2_ytbg4b3ebmxbwgbyycbmcewcfhchac23c4gcllctfcw3cyaczmdcgdeedfdd2wdladncdyeeadebledneexehde4tetg'
    points = data_trans.analysis_data(raw_data)
    points = data_trans.scale(points)
    draw(points)
