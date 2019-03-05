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
    raw_data = 'bdhbdabctbbybblbbebaxbahbab_zu_z2_zc_yx_ym_y4_yh_yh_yl_yw_zc_zh_znbaabagbambbdbb2bcabcfbctbdcbd4bdxbegbeybfdbfwbgebg4bhhb2hb2nb3db3mb3xb4cb4hb4mb4xblcblgbll_ut_un_un_ul_ul_u4_u2_uh_ug_ue_uc_ty_tt_t4_tf_ta_nt_n2_nf_nd_na_mx_mw_mu_mn_m4_m2_mf_me_md_mc_mc_mb_mb_ma_ly_ly_lx_lw_lu_ll_l2_lg_le_ld_lb_4x_4u_4n_44_4g_4c_aa_dy_en_hm_2e_3b_3x_lb_lw_my_nw_ua_we_yf_zwbcfbemb4ebnabubbwzbxzbzgcaccbacbxccnce2cf2cgfchachyc2mc34c4tclaclycmmcmncn4ctncuacuwcxdcxwcz2damdbudcwdegdged4g'
    points = data_trans.analysis_data(raw_data)
    points = data_trans.scale(points)
    draw(points)
