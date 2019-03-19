import matplotlib.pyplot as plt
from keras.models import load_model

from data_trans import get_velocity, scale, analysis_data
from data_vector import trans2vector_matrix
from load_mongodb import MongoData

model = load_model('./saved_model/weights25')
maxlen = 20
mongo_data = MongoData()
vps = mongo_data.get_mongodb_batch(size=2)
for vp in vps:
    # vp = '_cm_cw_df_dw_eg_ey_f4_ge_gw_h3_2a_2l_3a_33_3w_4d_44_lc_ln_me_mt_mu_mf_lx_l2_la_4n_4f_3y_4e_4n_le_lt_mg_mx_ng_nu_td_tm_ua_u2_uy_um_uc_t4_nz_n4_n3_tb_t3_tz_u3_uy_wl_xb_xm_ye_ytbecbd2bchbbgbaf_zf_ye_xe_we_ue_tg_n2_mn_lz_l2_4w_4f_32_2n_2a_hg_hf_hw_2g_2x_32_3y_42_4x_4h_3w_3c_23_hm_gx_gh_fu_ff_et_ef_dw_df_dw_eg_ez_fl_gb_ga_f2_fa_el_ec_dt_dd_cn_cd_bn_bf_aa_fb_gt_2h_4a_lm_ng_tz_wm_yg_zzbbmbdfbezbgmb2fb3zbnfbwmbzzcdgctzczzddgdgmd3zdnfdwmeawe4aengewmezcfc3ffwf3cftcfyggc3gfwg3dgtchm3huwhzc2c32fw2yg3bn3dg3ez3gn32g3ln3uw3zz4dg4gm'
    points = get_velocity(scale(analysis_data(vp)))
    while len(points) <= 20:
        vp = mongo_data.get_mongodb_batch(size=1)[0]
        points = get_velocity(scale(analysis_data(vp)))
    xs, ys = trans2vector_matrix(points, maxlen)
    plt.figure()
    x_axis = range(len(xs))
    y1 = []
    y11 = []
    y2 = []
    y22 = []

    y3 = []
    y33 = []
    for i, x in enumerate(xs):
        x = x.reshape(1, maxlen, 3)
        preds = model.predict(x, verbose=0)[0]
        y1.append(preds[0])
        y11.append(ys[i][0])
        y2.append(preds[1])
        y22.append(ys[i][1])
        y3.append(preds[2])
        y33.append(ys[i][2])
        print('prediction :%.4f' % preds[0] + '\t%.4f' % preds[1] + '\t%.4f' % preds[2])
        print('real       :%.4f' % ys[i][0] + '\t%.4f' % ys[i][1] + '\t%.4f' % ys[i][2])
    dis_a = [abs(y1[i] - y11[i]) for i in range(len(y1))]
    dis_t = [abs(y2[i] - y22[i]) for i in range(len(y2))]
    dis_v = [abs(y3[i] - y33[i]) for i in range(len(y3))]
    plt.subplot(141)
    plt.plot(x_axis, y1, label='prediction')
    plt.plot(x_axis, y11, label='real')
    plt.subplot(142)
    plt.plot(x_axis, y2, label='prediction')
    plt.plot(x_axis, y22, label='real')
    plt.subplot(143)
    plt.plot(x_axis, y3, label='prediction')
    plt.plot(x_axis, y33, label='real')
    plt.subplot(144)
    plt.plot(x_axis, dis_a, label='angle')
    plt.plot(x_axis, dis_t, label='time')
    plt.plot(x_axis, dis_v, label='velocity')
    plt.legend()
    plt.show()
