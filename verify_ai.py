import keras
from keras.models import load_model
import numpy as np
from data_trans import analysis_data
from traclus.main import check_trajectory
from vaptcha_feature import get_dataframe, calculate_velocity, calculate_angle


def get_prediction(vp):
    X = []
    df = get_dataframe(vp)
    df = calculate_velocity(df)
    if df is None:
        return 0
    df = calculate_angle(df)
    ql = df.angle_dif.quantile(0.75)
    tmp = df.angle_dif[df.angle_dif > ql]
    n = len(tmp)
    if n < 3:
        return 0
    start = tmp[:1].index.tolist()[0]
    end = tmp[-1:].index.tolist()[0]
    if start < 6:
        start = 6
    if len(df) - end < 3:
        end -= 2
    start_df = df[2:start]
    end_df = df[end:]
    X.append(np.asarray(
        [df.v.mean(), df.v.max(), df.a.mean(), start_df.v.mean(), start_df.a.mean(), end_df.v.mean(),
         end_df.a.mean()]))
    X = np.asarray(X)

    model = keras.models.load_model('/Users/sunyuning/PycharmProjects/validation/dbscan.h5')
    p = list(model.predict(X))[0][0] * 0.8

    p_rate = check_trajectory(vp)
    p = (p_rate - 0.5) * 0.2 + p
    return str(p)


if __name__ == '__main__':
    vp = '_el_fa_f4_gc_gt_hf_2f_2t_3d_3x_44_la_l3_lw_mc_mh_mn_nb_n2_nu_ta_t2_tw_uc_u2_ut_uy_wb_wd_wf_wh_wh_wf_wc_uw_u2_ua_t4_td_nx_nl_ne_mw_mh_lz_ll_le_4z_4t_4f_4b_3w_3l_3g_3b_2w_23_2b_hm_hh_mt_mz_ne_nh_n3_nl_nu_nx_ta_tc_td_tf_th_t2_t3_t3_t3_t2_t3_tn_tw_ua_uc_ue_uf_u3_un_uy_we_w4_wy_xg_xt_yb_y3_yy_ze_z3_zl_zm_zm_zm_zl_z3_zg_zc_yz_yw_yt_y3_yg_ye_yc_xy_xw_xn_xl_xl_xl_xl_aa_c2_e3_ft_gw_2e_32_3x_4g_4n_ml_mz_ne_tb_ud_uw_wm_y2babbbwbd2bfebgwb24b4eblzbnmbuebwzbyncahccdccycdtcemcf4cgfchcchyc2wc3nc4lcl2cmdcnacnxctucumcwhcydcyzczxdandchdftd2hd4ydmldtedwu'
    print(get_prediction(vp))
