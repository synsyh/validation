import json
import os
import csv
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# 测试数据时使用的file名字
# filename = 'test.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
# jsonfilename = 'test.json'
# csvfilename = 'test.json'

# 读取json文件
def read_json(filename):
    f = open(filename, encoding='utf-8')
    result = json.load(f)
    return result


# 将列表中的字典进行转化
def transform_dic_(L):
    B = []
    for i in range(len(L)):
        a = L[i]['x']
        b = L[i]['y']
        c = L[i]['t']  # 单位毫秒ms
        B.append([a, b, c])
    result = list(map(lambda _item: (','.join(map(lambda x: str(x), _item))), B))
    result = ';'.join(result)
    return result


def transform_dic(L):
    B = []
    # L = eval(L)
    for i in range(len(L)):
        a = L[i]["x"]
        b = L[i]["y"]
        c = L[i]["time"]  # 单位毫秒ms
        B.append([a, b, c])
    result = list(map(lambda _item: (','.join(map(lambda x: str(x), _item))), B))
    result = ';'.join(result)

    return result


# 将txt文件内容转为json
def exact_edges(txtpath, jsonpath):
    txtopen = open(txtpath, 'r')
    file = txtopen.read().splitlines()
    list_data = []
    for i in range(len(file)):
        list_data.append(transform_dic_(eval(file[i])))  # 先将轨迹由str转为list,然后对列表中的字典进行转化，以便后续处理
        # list_data.append(eval(file[i]))
    with open(jsonpath, 'w', encoding='utf-8') as json_file:
        json.dump(list_data, json_file, ensure_ascii=False)


# 原始数据处理
def data_process(data):
    # data['point'] = data['point'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[1:-1]])
    data['point'] = data['point'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[1:]])
    # 提取 x坐标 y坐标 t 目标点x坐标  目标点y坐标（起始点作为目标点）
    df = pd.DataFrame()

    df['x'] = data['point'].apply(lambda x: np.array(x)[:, 0])
    df['y'] = data['point'].apply(lambda x: np.array(x)[:, 1])
    df['t'] = data['point'].apply(lambda x: np.array(x)[:, 2])
    df['target_x'] = data['point'].apply(lambda x: np.array(x)[0, 0])
    df['target_y'] = data['point'].apply(lambda x: np.array(x)[0, 1])

    return df


# 差分处理
def data_diff(data, name_list):
    for name in name_list:
        data['diff_' + name] = data[name].apply(lambda x: pd.Series(x).diff().dropna().tolist())  # 列表前后值做差分处理
        data['diff_' + name] = data['diff_' + name].apply(lambda x: [0] if x == [] else x)  # !!注意 一个点的情况
    return data


def abs_data_diff(data, name_list):
    for name in name_list:
        data['abs_' + name] = data[name].apply(lambda x: pd.Series(x).abs().dropna().tolist())  # 列表前后值做差分处理
        data['abs_' + name] = data['abs_' + name].apply(lambda x: [0] if x == [] else x)  # !!注意 一个点的情况
    # print(data['abs_' + name])
    return data


# 轨迹偏移点角度的计算方程
def calculate_angle(x_value, y_value):  # 计算转移角度值，用以确定曲线的拐点
    Angle = []
    for i in range(1, len(x_value) - 1):
        angle_line_b = np.sqrt((x_value[i + 1] - x_value[i - 1]) ** 2 + (y_value[i + 1] - y_value[i - 1]) ** 2)
        angle_line_a = np.sqrt((x_value[i + 1] - x_value[i]) ** 2 + (y_value[i + 1] - y_value[i]) ** 2)
        angle_line_c = np.sqrt((x_value[i] - x_value[i - 1]) ** 2 + (y_value[i] - y_value[i - 1]) ** 2)
        value = (angle_line_a ** 2 + angle_line_c ** 2 - angle_line_b ** 2) / (
                (2 * angle_line_a * angle_line_c) + 0.00001)
        if value <= -1:  # 计算过程中由于四舍五入的原因，导致Value会小于-1，此时加入判断让其为-1
            value = -1

        if value >= 1:  # 计算过程中由于四舍五入的原因，导致Value会小于-1，此时加入判断让其为-1
            value = 1
        Angle.append(np.arccos(value) * 180 / np.pi)

    return Angle


# 获取轨迹相邻采样点偏移角度数据
def get_angle(data):  # 计算转移角度值，提取轨迹的曲线及拐点特征
    Angle = []
    # dfGroup = pd.DataFrame()
    for x, y in zip(data['x'], data['y']):
        Angle.append((calculate_angle(x, y)))

    data['Angle'] = Angle

    return data


# 获取距离数据
def get_dist(data):
    dist_target = []
    dist = []  # 间隔距离值
    dist_x_target = []
    dist_y_target = []
    dist_all = []  # 所有间隔距离的依次加和值
    diff_distanc = 0
    # 各点与起始点（目标点）的距离
    for x, y, target_x, target_y in zip(data['x'], data['y'], data['target_x'], data['target_y']):
        dist_target.append(np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2))
    # 两点之间的距离
    for x, y in zip(data['diff_x'], data['diff_y']):
        dist_value = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
        # dist.append(np.sqrt(np.array(x) ** 2 + np.array(y) ** 2))
        dist.append(dist_value)
        summary_dis = 0
        diff_distanc = [0]
        for i in dist_value:
            summary_dis += i
            diff_distanc.append(summary_dis)

        dist_all.append(diff_distanc)

    # 各点x坐标与目标点x坐标的距离
    for x, target_x in zip(data['x'], data['target_x']):
        dist_x_target.append(np.sqrt((x - target_x) ** 2))
    # 各点y坐标与目标点y坐标的距离
    for y, target_y in zip(data['y'], data['target_y']):
        dist_y_target.append(np.sqrt((y - target_y) ** 2))

    data['dist_target'] = dist_target
    data['dist'] = dist
    data['dist_all'] = dist_all
    data['dist_x_target'] = dist_x_target
    data['dist_y_target'] = dist_y_target
    # print(dist_all)

    return data


# 获取速度数据
def get_v(data):
    v = []
    v_x = []
    v_y = []
    # 获取两点之间的速度
    for dist, t in zip(data['dist'], data['diff_t']):
        v0 = dist / t
        v0 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, v0))  # !! 注意除数为0的情况
        v.append(v0)
    # 获取两点x坐标之间的速度
    for x, t in zip(data['diff_x'], data['diff_t']):
        v1 = np.array(x) / np.array(t)
        v1 = list(map(lambda x: 0 if x == np.inf or x == -np.inf or np.isnan(x) else x, v1))
        v_x.append(v1)
    # 获取两点y坐标之间的速度
    for y, t in zip(data['diff_y'], data['diff_t']):
        v2 = np.array(y) / np.array(t)
        v2 = list(map(lambda x: 0 if x == np.inf or x == -np.inf or np.isnan(x) else x, v2))
        v_y.append(v2)

    data['v'] = v
    data['v_x'] = v_x
    data['v_y'] = v_y

    return data


# 获取加速度数据
def get_a(data):
    a = []
    a_x = []
    a_y = []
    abs_a = []
    abs_a_x = []
    abs_a_y = []
    # 获取两点之间的加速度
    for v, t in zip(data['diff_v'], data['diff_t']):
        v = np.array(v)
        t = np.array(t)
        a_t = (t[:-1] + t[1:]) / 2
        a0 = v / a_t
        a0 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, a0))  # !! 注意除数为0的情况
        # !!注意 列表为空
        if a0 == []:
            a0 = [0]
        a.append(a0)
        # 获取两点x坐标之间的加速度
    for v_x, t in zip(data['diff_v_x'], data['diff_t']):
        v_x = np.array(v_x)
        t = np.array(t)
        a_t = (t[:-1] + t[1:]) / 2
        a1 = v_x / a_t
        a1 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, a1))  # !! 注意除数为0的情况
        if a1 == []:
            a1 = [0]
        a_x.append(a1)
        # 获取两点x坐标之间的加速度
    for v_y, t in zip(data['diff_v_y'], data['diff_t']):
        v_y = np.array(v_y)
        t = np.array(t)
        a_t = (t[:-1] + t[1:]) / 2
        a2 = v_y / a_t
        a2 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, a2))  # !! 注意除数为0的情况
        if a2 == []:
            a2 = [0]
        a_y.append(a2)

    for v, t in zip(data['abs_diff_v'], data['diff_t']):
        v = np.array(v)
        t = np.array(t)
        a_t = (t[:-1] + t[1:]) / 2
        a0 = v / a_t
        a0 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, a0))  # !! 注意除数为0的情况
        # !!注意 列表为空
        if a0 == []:
            a0 = [0]
        abs_a.append(a0)
        # 获取两点x坐标之间的加速度
    for v_x, t in zip(data['abs_diff_v_x'], data['diff_t']):
        v_x = np.array(v_x)
        t = np.array(t)
        a_t = (t[:-1] + t[1:]) / 2
        a1 = v_x / a_t
        a1 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, a1))  # !! 注意除数为0的情况
        if a1 == []:
            a1 = [0]
        abs_a_x.append(a1)
        # 获取两点x坐标之间的加速度
    for v_y, t in zip(data['abs_diff_v_y'], data['diff_t']):
        v_y = np.array(v_y)
        t = np.array(t)
        a_t = (t[:-1] + t[1:]) / 2
        a2 = v_y / a_t
        a2 = list(map(lambda x: 0 if x == np.inf or x == -np.inf else x, a2))  # !! 注意除数为0的情况
        if a2 == []:
            a2 = [0]
        abs_a_y.append(a2)

    data['a'] = a
    data['a_x'] = a_x
    data['a_y'] = a_y
    data['abs_a'] = abs_a
    data['abs_a_x'] = abs_a_x
    data['abs_a_y'] = abs_a_y

    return data


# 按距离进行划分，若前n段距离加和刚和满足条件，则跳出循环，且返回满足条件时在列表中的位置下标
def mean_point_place_1(dis_L, length):
    a = []
    # dis_L = get_length(P)
    for i in range(len(dis_L) - 1):
        if dis_L[i] <= length < dis_L[i + 1]:
            a.append(i)
            break
    return a[0]


# 按照等距离进行划分，获取与原数据点数相同的等距数据点
def get_mean_point(data):
    mean_point_x = []
    mean_point_y = []

    for x, y, z in zip(data['x'], data['y'], data['dist_all']):
        total_length = z[-1]
        L_div_num = total_length / len(x)
        point_list_x = []
        point_list_y = []
        for i in range(len(x)):
            length = L_div_num * (i)
            k = mean_point_place_1(z, length)
            a = length - z[k]
            tan_angle = (y[k + 1] - y[k]) / (x[k + 1] - x[k] + 0.00001)
            angle = np.arctan(tan_angle)
            if x[k + 1] - x[k] >= 0:
                z_x = a * np.cos(angle) + x[k]
                z_y = a * np.sin(angle) + y[k]
            else:
                z_x = x[k] - a * np.cos(angle)
                z_y = y[k] - a * np.sin(angle)
            point_list_x.append(z_x)
            point_list_y.append(z_y)
        point_list_x.append(x[-1])
        point_list_y.append(y[-1])
        mean_point_x.append(point_list_x)
        mean_point_y.append(point_list_y)

    data['mean_point_x'] = mean_point_x
    data['mean_point_y'] = mean_point_y

    return data


# 获取轨迹相邻采样点偏移角度数据
def get_mean_point_angle(data):  # 计算转移角度值，提取轨迹的曲线及拐点特征
    Angle = []
    for x, y in zip(data['mean_point_x'], data['mean_point_y']):
        Angle.append((calculate_angle(x, y)))

    data['mean_point_angle'] = Angle

    return data


# 获取轨迹在拐点时的变化数据（以角度最小值和速度小值以及时间间隔大值来共同确定）#暂时只提取弧度变化最大时候的拐点特征
def get_turning_change1(data):  # 暂时以该方式进行拐点特征提取，后期进行改进
    # 获取轨迹的拐点位置列表
    turning_point_v_list = []
    point_index_list = []
    for x, y, z, u in zip(data['Angle'], data['v'], data['diff_t'], data['dist_all']):
        point_index = []
        x_list = sorted(x)
        point_index_value = find_list_index_1(x, x_list[0])[0]
        if point_index_value <= 3 or (len(x) - point_index_value) <= 4:  # 首先判断拐点是否在轨迹的首尾处，以首尾3个数区隔开
            point_index_value = find_list_index_1(x, x_list[1])[0]
            point_index.append(point_index_value)
        else:  # 若偏移点所在位置位于轨迹中间段
            point_index.append(point_index_value)  # 将该点添加到偏移点位置列表

        point_v = min(y[point_index_value], y[point_index_value + 1])
        turning_point_v_list.append(point_v)

    data['turning_point_v'] = turning_point_v_list

    return data


# 获取指定值在列表中的位置下标（从0开始）
def find_list_index_1(arr, item):
    return [i for i, a in enumerate(arr) if a == item]


# 获取指定值在列表中的位置下标（从1开始）
def find_list_index(arr, item):
    return [i + 1 for i, a in enumerate(arr) if a == item]


# 获取轨迹在绘制过程中的来回滑动（坐标重复）次数（以单个轴向上轨迹滑动次数确定）
def get_back_num(List):
    b = []
    for i in range(0, len(List)):
        c = 1 if List[i] > 0 else 0
        b.append(c)

    diff_back = pd.Series(b).diff().dropna().tolist()  # 前后值做差分处理生成差值列表，如果值为1或者-1，则表示为轴向转点
    diff_back_max = find_list_index(diff_back, -1)
    diff_back_min = find_list_index(diff_back, 1)
    back_num = len(diff_back_max) + len(diff_back_min)

    return back_num


# 对数据进行相应的特征提取操作：首尾值，最大值，最小值，绝对值的最大值，正负峰间值，平均值，标准差，四分位数，峰度与斜度
# 提取序列中的首尾值
def get_feature_start_end(data, name, n=1):  # 可以提取序列中的n个值
    dfGroup = pd.DataFrame()
    dfGroup[name + '_start'] = data.apply(lambda x: np.mean(x[:n]))  # 获取起始值
    dfGroup[name + '_end'] = data.apply(lambda x: np.mean(x[len(x) - n:]))  # 获取结束值

    return dfGroup


# 提取序列中的前后15个值的均值与方差
def get_feature_start_end_mean_std(data, name, n=15):  # 可以取序列中的n个值,默认前后15个点
    dfGroup = pd.DataFrame()
    dfGroup[name + '_start_mean'] = data.apply(lambda x: np.mean(x[:n]))  # 获取前n个值的均值
    dfGroup[name + '_end_mean'] = data.apply(lambda x: np.mean(x[len(x) - n:]))  # 获取后n个值的均值
    dfGroup[name + '_start_std'] = data.apply(lambda x: np.std(x[:n]))  # 获取前n个值的方差
    dfGroup[name + '_end_std'] = data.apply(lambda x: np.std(x[len(x) - n:]))  # 获取后n个值的方差

    return dfGroup


# 获取加速度的前n个值均值，然后与0比较
def get_a_value_change(data, name, n=2):  # 可以提取序列中的n个值
    dfGroup = pd.DataFrame()
    # dfGroup[name + '_start_compare_zero'] = data.apply(lambda x: 1 if np.mean(x[:n]) >0 else 0)
    # #获取加速度的前n个值均值，然后与0比较，小于0取值为1（正常画之前速度会从0开始增加，则加速度为正值。结束时，速度会变慢，则加速度为负值）
    # dfGroup[name + '_end_compare_zero'] = data.apply(lambda x: 1 if np.mean(x[len(x)- n:]) <0 else 0)
    dfGroup[name + '_start_end_compare_zero'] = data.apply(
        lambda x: 1 if np.mean(x[:n]) > 0 and np.mean(x[len(x) - n:]) < 0 else 0)
    # 获取加速度的前n个值均值，然后与0比较，小于0取值为1（正常画之前速度会从0开始增加，则加速度为正值。结束时，速度会变慢，则加速度为负值）

    return dfGroup


# 获取序列的最大最小值
def get_feature_max_min(data, name):
    dfGroup = pd.DataFrame()
    # if abs_cal ==False:
    dfGroup[name + '_max'] = data.apply(lambda x: max(x))  # 获取最大值
    dfGroup[name + '_min'] = data.apply(lambda x: min(x))  # 获取最小值

    return dfGroup


# 获取序列的峰谷值差（最大值减去最小值）
def get_feature_ptp(data, name):
    dfGroup = pd.DataFrame()
    dfGroup[name + '_ptp'] = data.apply(lambda x: max(x)).sub(data.apply(lambda x: min(x)))
    # 最大值减去最小值，peak-to-peak value 正负峰间值

    return dfGroup


# 获取序列的绝对值的峰谷值差（最大值减去最小值）
def get_feature_abs_max(data, name):
    dfGroup1 = pd.DataFrame()
    dfGroup = get_feature_max_min(data, name)
    abs_max = []
    for max_, min_ in zip(dfGroup[name + '_max'], dfGroup[name + '_min']):
        abs_max.append(max(abs(max_), abs(min_)))

    dfGroup1[name + '_abs_max'] = abs_max

    return dfGroup1


# 获取序列的均值方差
def get_feature_mean_std(data, name, cv_cal=False):
    dfGroup = pd.DataFrame()
    dfGroup[name + '_mean'] = data.apply(lambda x: np.mean(x))  # 均值
    dfGroup[name + '_std'] = data.apply(lambda x: np.std(x))  # 标准差，若求方差及为标准差的平方
    if cv_cal == True:
        dfGroup[name + '_cv'] = dfGroup[name + '_std'].div(dfGroup[name + '_mean'], fill_value=0)  # 变异系数，cv=std/mean
        dfGroup[name + '_cv'] = dfGroup[name + '_cv'].replace([np.inf, -np.inf], [0, 0])
        dfGroup[name + '_cv'] = dfGroup[name + '_cv'].fillna(0)

    return dfGroup


# 四分位差
def get_feature_q1_q3(data, name):
    dfGroup = pd.DataFrame()
    dfGroup[name + '_Q1'] = data.apply(lambda x: np.percentile(x, 0.25))  # 求四分位数,0.25为下分位数，0.5为中位数，0.75为上分位数
    dfGroup[name + '_Q2'] = data.apply(lambda x: np.percentile(x, 0.5))
    dfGroup[name + '_Q3'] = data.apply(lambda x: np.percentile(x, 0.75))
    dfGroup[name + '_interRan'] = dfGroup[name + '_Q3'].sub(dfGroup[name + '_Q1'])  # Q3-Q1为四分位差，四分位差也是衡量数据的发散程度的指标之一

    return dfGroup


def get_feature_skew_kurt(data, name):
    dfGroup = pd.DataFrame()
    dfGroup[name + '_skew'] = data.apply(lambda x: pd.Series(x).skew()).fillna(0)
    # 斜度，三阶中心距除以标准差的三次方，表示随机变量与中心分布的不对称程度，向右倾斜值为正，向左值为负
    dfGroup[name + '_kurt'] = data.apply(lambda x: pd.Series(x).kurt()).fillna(0)
    # 峰度，概率密度在均值处峰值高低的特征，常定义四阶中心矩除以方差的平方，减去三。随机变量在均值附近的相对平坦程度或峰值程度，以正态分布为界，峰度值为0，如比正态分布陡，峰度值大于0，否则小于0.

    return dfGroup


# 对点的位置进行特征提取
def get_point_feature(df):
    # 点的位置进行特征提取,点的特征只需要提取x或y轴上的峰谷差值
    point_x_ptp = get_feature_ptp(df['x'], 'x')  # 提取轨迹在x轴上的坐标差值（x值的最大最小值差）
    point_y_ptp = get_feature_ptp(df['y'], 'y')  # 提取轨迹在y轴上的坐标差值（y值的最大最小值差）
    point = pd.concat([point_x_ptp, point_y_ptp], axis=1)

    return point


# 对角值进行特征提取
def get_angle_feature(df):
    # 加速度进行特征提取，后期如有需要，可以考虑继续提取加速度在x,y轴上的分量特征
    # a_max_min = get_feature_max_min(df['a'], 'a')  # 提取加速度的最大最小特征，现以加速度绝对值代替。
    a_abs_max = get_feature_abs_max(df['a'], 'a')  # 提取加速度的绝对值的最大值（忽略加速度的方向）
    a_ptp = get_feature_ptp(df['a'], 'a')  # 提取加速度峰谷特征（速度的最大最小值差）
    a_mean_std = get_feature_mean_std(df['a'], 'a')  # 提取加速度的均值和标准差特征

    angle = pd.concat([a_abs_max, a_ptp, a_mean_std], axis=1)

    return angle


# 获取轨迹在拐点时的变化数据（以角度最小值和速度小值以及时间间隔大值来共同确定）#暂时只提取弧度变化最大时候的拐点特征
def get_turning_change(data):  # 暂时以该方式进行拐点特征提取，后期进行改进
    # 获取轨迹的拐点位置列表
    dfGroup = pd.DataFrame()
    turning_point_v_list = []
    for x, y, z, u in zip(data['Angle'], data['v'], data['diff_t'], data['a']):
        point_index = []
        x_list = sorted(x)
        point_index_value = find_list_index_1(x, x_list[0])[0]
        if point_index_value <= 3 or (len(x) - point_index_value) <= 4:  # 首先判断拐点是否在轨迹的首尾处，以首尾3个数区隔开
            point_index_value = find_list_index_1(x, x_list[1])[0]
            point_index.append(point_index_value)
        else:  # 若偏移点所在位置位于轨迹中间段
            point_index.append(point_index_value)  # 将该点添加到偏移点位置列表

        point_v = min(y[point_index_value], y[point_index_value + 1])
        point_angle = x[point_index_value]
        point_a = u[point_index_value]

        turning_point_v_list.append(point_v)

    data['turning_point_v'] = turning_point_v_list

    return data


def get_ptoline_distance(x1, y1, x2, y2, x, y):
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)  # http://blog.csdn.net/angelazy/article/details/38489293
    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    r = cross / d2

    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    distacne = np.sqrt((x - px) * (x - px) + (py - y) * (py - y))
    return distacne


# 获取轨迹的拐点位置（以数学方式即向量求点距离最大值来确定）
def get_turning_point1(data):  # 暂时以该方式进行拐点特征提取，后期进行改进
    # 获取轨迹的拐点位置列表
    dfGroup = pd.DataFrame()
    turning_point = []
    min_angle = []
    point_angle = []
    for x, y, z, u in zip(data['x'], data['y'], data['diff_t'], data['dist_all']):
        dis_list = []
        point_list = []
        dis_list_2 = []
        for i in range(len(x)):
            dis = get_ptoline_distance(x[0], y[0], x[-1], y[-1], x[i], y[i])
            dis_list.append(dis)
        p_ind = dis_list.index(max(dis_list))
        point_list.append(p_ind)

        for i in range(p_ind):
            dis = get_ptoline_distance(x[0], y[0], x[p_ind], y[p_ind], x[i], y[i])
            dis_list_2.append(dis)
        for i in range(p_ind, len(x)):
            dis = get_ptoline_distance(x[p_ind], y[p_ind], x[-1], y[-1], x[i], y[i])
            dis_list_2.append(dis)

        p_ind_2 = dis_list_2.index(max(dis_list_2))
        point_list.append(p_ind_2)

        turning_point.append(sorted(point_list))
    # print(turning_point)
    data['turning_point_dis'] = turning_point
    # dfGroup['min_angle'] = min_angle
    return data


def calcul_true_angle(x1, y1, x2, y2, x, y):
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    d2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    # if cross>0:
    angle = np.arccos(cross / (d2 * d1)) * 180 / np.pi
    # else:
    #     angle =180 -np.arccos(cross/(d2*d1))*180/np.pi
    return angle


def calcul_f_b_mean_angle(x, y, x1, y1, x2, y2):
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    d2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    if y1 <= y:
        angle = np.arccos(cross / (d2 * d1)) * 180 / np.pi
    else:
        angle = -np.arccos(cross / (d2 * d1)) * 180 / np.pi
    return angle


# 按距离进行划分，若前n段距离加和刚和满足条件，则跳出循环，且返回满足条件时在列表中的位置下标
def front_point_place(dis_L, length):
    a = []
    # dis_L = get_length(P)
    if len(dis_L) > 16:
        dis_L = dis_L[0:int(len(dis_L) / 2)]

    if min(dis_L) > length:
        a.append(1)

    elif max(dis_L) <= length:
        a.append(len(dis_L) - 1)

    else:
        for i in range(len(dis_L) - 1):
            if dis_L[i] <= length < dis_L[i + 1]:
                a.append(i + 2)
                break
    if len(a) == 0:
        a.append(1)

    return a[0]


def back_point_place(dis_L, length):
    a = []
    # dis_L = get_length(P)
    if max(dis_L) < length:
        a.append(len(dis_L) - 1)
    elif min(dis_L) > length:
        a.append(1)
    else:
        for i in range(len(dis_L) - 1):
            if dis_L[i] <= length < dis_L[i + 1]:
                a.append(i + 1)
                break
    if len(a) == 0:
        a.append(1)

    return a[0]


def find_repeat(source, elmt):  # The source may be a list or string.
    elmt_index = []
    s_index = 0
    e_index = len(source)
    while (s_index < e_index):
        try:
            temp = source.index(elmt, s_index, e_index)
            elmt_index.append(temp)
            s_index = temp + 1
        except ValueError:
            break

    return elmt_index


def get_radius_point(x, y, p_x, p_y, leng):
    # p_index_y =list(y).index(p_y)
    x_index = find_repeat(list(x), p_x)
    y_index = find_repeat(list(y), p_y)

    p_index = [l for l in x_index if l in y_index][0]
    x_list = x[::-1][-p_index:]
    y_list = y[::-1][-p_index:]

    front_list = []
    back_list = []

    for i in range(p_index):
        dist = np.sqrt((p_x - x_list[i]) ** 2 + (p_y - y_list[i]) ** 2)
        # dist =
        front_list.append(dist)

    for i in range(p_index, len(x)):
        dist = np.sqrt((p_x - x[i]) ** 2 + (p_y - y[i]) ** 2)
        back_list.append(dist)

    k1 = front_point_place(front_list, leng)
    k2 = back_point_place(back_list, leng)

    # return front_list,front_list
    return p_index - k1, k2 + p_index


# 获取轨迹的拐点位置（以角度最小值和速度小值以及时间间隔大值来共同确定）#暂时只提取弧度变化最大的一个拐点特征
def get_turning_point(data):  # 暂时以该方式进行拐点特征提取，后期进行改进
    # 获取轨迹的拐点位置列表
    dfGroup = pd.DataFrame()
    turning_point = []  # 记录拐点位置的序列
    point_angle = []  # 记录拐点的偏转角（手绘满足要求的偏转角），记录两个点时使用
    true_angle = []  # 记录拐点的实际夹角，记录两个点时使用
    turning_point_loc = []  # 记录拐点的x,y值（位置）
    turning_point_index = []  # 记录拐点的顺序值（若只有一个拐点取，若两个拐点，取其序号值）
    point_fro_back_index = []  # 记录拐点前后满足要求的点的序列（固定半径作为范围）
    clockwise_direction = []  # 记录拐点出轨迹走势的顺手方向（当在一二象限，顺时针为顺手，三四象限，逆时针为顺手）
    abs_angle = []
    true_angle_list = []  # 记录单个点时使用
    point_angle_list = []
    # 获取相应序列范围内的特征，包括速度，加速度，间隔时间等
    v_mean_point_list = []  # 记录拐点前后满足要求的速度均值
    v_std_point_list = []  # 记录拐点前后满足要求的速度标准差
    a_mean_point_list = []  # 记录拐点前后满足要求的加速度均值
    a_std_point_list = []
    dis_mean_point_list = []  # 记录拐点前后满足要求的间隔距离均值
    dis_std_point_list = []
    # v_mean_point = []                         # 记录拐点前后满足要求的速度均值
    # v_std_point =[]                         #记录拐点前后满足要求的速度标准差
    # a_mean_point = []                       # 记录拐点前后满足要求的加速度均值
    # a_std_point = []
    # dist_mean_point = []                    # 记录拐点前后满足要求的间隔距离均值
    # dist_std_point = []

    for x, v, a, t, dist, n, point_x, point_y in zip(data['Angle'], data['v'], data['abs_a'], data['diff_t'],
                                                     data['dist'],
                                                     data['turning_point_dis'], data['x'], data['y']):
        # 获取拐点的位置
        point_index = []
        point_index_all = []
        # point_angle_list = []
        x_list = sorted(x)

        # print(x)

        point_index_value_0 = find_list_index_1(x, x_list[0])[0]
        point_index_value_1 = find_list_index_1(x, x_list[1])[0]
        point_index_value_2 = find_list_index_1(x, x_list[2])[0]
        point_index_value_3 = find_list_index_1(x, x_list[3])[0]
        point_index_value_4 = find_list_index_1(x, x_list[4])[0]

        five_point_list = [point_index_value_0, point_index_value_1, point_index_value_2, point_index_value_3,
                           point_index_value_4]

        point_list = []
        # 首先判断拐点是否在轨迹的首尾处，以首尾3个数区隔开,需要偏移点在中心位置，接着再判断每个候选点是否与角度平均值相差在12°之内
        for i in five_point_list:
            if i > 2 and (len(x) - i) > 3 and abs(np.mean(x) - x[i]) > 15:
                point_list.append(i)
        if len(point_list) > 0:
            point_index.append(point_list[0])  # 将满足要求的转角弧度最大的点添加到列表
            point_index_all.append(point_list[0])
            # point_angle_list.append(x[point_list[0]])      #将满足要求的转角点的角度值添加到列表
            for i in range(1, len(point_list)):
                if abs(point_list[i] - point_list[0]) >= 4:
                    # point_index.append(point_list[i])         #前后两个拐点要求间隔大于等于4，将满足要求的转角弧度第二的点添加到列表
                    point_index_all.append(point_list[i])
                    # point_angle_list.append(x[point_list[i]]) #将满足要求的转角点的角度值添加到列表
                    break
        elif len(point_list) == 1:
            point_index.append(point_list[0])
            point_index_all.append(point_list[0])
            # point_angle_list.append(x[point_list[0]])
            # for nu in n:
            #     if abs(point_list[0]-nu)>4:
            #         point_index.append(nu-1)
        else:
            point_index.append(n[0])
            point_index_all.append(n[0])
            # point_index.append(n[1])
        # turning_point.append(sorted(point_index))
        turning_point.append(point_index[0])

        # 获取拐点在满足要求的点列表中的序号
        point_index_num = find_list_index_1(sorted(point_index_all), point_index[0])
        # print(point_index_num)
        turning_point_index.append(point_index_num[0])

        # 获取拐点前后相应范围内的满足要求的点的序列
        # true_angle_list = []
        point_loc_list = []
        f_back_list = []
        point_index = sorted(point_index)  # 将拐点序列列表排序

        for i in point_index:  # 遍历拐点序列，将对应位置的轨迹偏移角度取出
            point_angle_list.append(x[i])

        for i in point_index:  # 遍历拐点序列，以拐点为圆心，半径length来提取相应的特征
            # 首先获取拐点相应半径的起始点序号和结束序号，以便后面获取速度，加速度等特征
            i_num = 4  # 以首尾5个点作为判断，来计算出拐点的实际坐标位置
            length = 40  # 计算以拐点为圆心，半径为length的范围内的特征值
            if i <= i_num:
                point_1_x, point_1_y = point_x[0], point_y[0]
                point_2_x, point_2_y = point_x[i + i_num], point_y[i + i_num]
                dis_list = []
                for j in range(0, i + i_num):
                    dis = get_ptoline_distance(point_1_x, point_1_y, point_2_x, point_2_y, point_x[j], point_y[j])
                    dis_list.append(dis)
                # p_ind = dis_list.index(max(dis_list))
                # angle = calcul_true_angle(point_1_x, point_1_y, point_2_x, point_2_y, point_x[p_ind], point_y[p_ind])
                # point_angle_list.append(angle)
                p_ind = dis_list.index(max(dis_list))
                p_ind_x = point_x[0: i + i_num][p_ind]
                p_ind_y = point_y[0: i + i_num][p_ind]
                k1, k2 = get_radius_point(point_x, point_y, p_ind_x, p_ind_y, length)
                angle = calcul_true_angle(p_ind_x, p_ind_y, point_x[k2], point_y[k2], point_x[k1], point_y[k1])
                true_angle_list.append(angle)
                point_loc_list.append([p_ind_x, p_ind_y])
                f_back_list.append([k1, k2])
            elif len(point_x) - i <= i_num:
                point_1_x, point_1_y = point_x[-1], point_y[-1]
                point_2_x, point_2_y = point_x[i - i_num], point_y[i - i_num]
                dis_list = []
                for j in range(i - i_num, len(point_x)):
                    dis = get_ptoline_distance(point_1_x, point_1_y, point_2_x, point_2_y, point_x[j], point_y[j])
                    dis_list.append(dis)
                p_ind = dis_list.index(max(dis_list))
                p_ind_x = point_x[i - i_num: len(point_x)][p_ind]
                p_ind_y = point_y[i - i_num: len(point_x)][p_ind]
                k1, k2 = get_radius_point(point_x, point_y, p_ind_x, p_ind_y, length)
                angle = calcul_true_angle(p_ind_x, p_ind_y, point_x[k2], point_y[k2], point_x[k1], point_y[k1])
                # angle = calcul_true_angle(point_1_x, point_1_y, point_2_x, point_2_y, p_ind_x, p_ind_y)
                true_angle_list.append(angle)
                point_loc_list.append([p_ind_x, p_ind_y])
                f_back_list.append([k1, k2])
            else:
                point_1_x, point_1_y = point_x[i - i_num], point_y[i - i_num]
                point_2_x, point_2_y = point_x[i + i_num], point_y[i + i_num]
                dis_list = []
                for j in range(i - i_num, i + i_num + 1):
                    dis = get_ptoline_distance(point_1_x, point_1_y, point_2_x, point_2_y, point_x[j], point_y[j])
                    dis_list.append(dis)
                p_ind = dis_list.index(max(dis_list))
                p_ind_x = point_x[i - i_num: i + i_num][p_ind]
                p_ind_y = point_y[i - i_num: i + i_num][p_ind]

                # print([point_x,point_y,p_ind_x,p_ind_y])

                k1, k2 = get_radius_point(point_x, point_y, p_ind_x, p_ind_y, length)
                angle = calcul_true_angle(p_ind_x, p_ind_y, point_x[k2], point_y[k2], point_x[k1], point_y[k1])
                true_angle_list.append(angle)
                point_loc_list.append([p_ind_x, p_ind_y])
                f_back_list.append([k1, k2])

        for i in point_loc_list:
            front_index = f_back_list[0][0]
            back_index = f_back_list[0][1]
            front_point_loc = [point_x[front_index], point_y[front_index]]
            back_point_loc = [point_x[back_index], point_y[back_index]]
            f_b_mean_loc = [int(point_x[front_index] / 2 + point_x[back_index] / 2),
                            int(point_y[front_index] / 2 + point_y[back_index] / 2)]

            angle = calcul_f_b_mean_angle(f_b_mean_loc[0], f_b_mean_loc[1], i[0], i[1], i[0] + 10, i[1])
            front_mean = [f_b_mean_loc[0] - front_point_loc[0], f_b_mean_loc[1] - front_point_loc[1]]
            back_mean = [f_b_mean_loc[0] - back_point_loc[0], f_b_mean_loc[1] - back_point_loc[1]]
            # 叉积大于0，则为逆时针；叉积小于0，则为顺时针
            cross_product = front_mean[0] * back_mean[1] - front_mean[1] * back_mean[0]

            # print(angle)
            # print(cross_product)
            if 0 <= angle < 160 and cross_product > 0:  # 若在一二象限，且为逆时针，则为顺手方向
                clockwise_direction.append(1)
            elif -160 <= angle < 0 and cross_product < 0:  # 若在三四象限，且为顺时针，则为顺手方向
                clockwise_direction.append(1)
            else:  # 其余情况，都为反手方向
                clockwise_direction.append(0)

        # print(clockwise_direction)

        f_back_v_list = v[f_back_list[0][0]:f_back_list[0][1] + 1]
        f_back_a_list = a[f_back_list[0][0]:f_back_list[0][1] + 1]
        # f_back_t_list = t[f_back_list[0][0]:f_back_list[0][1]+1]
        f_back_dist_list = dist[f_back_list[0][0]:f_back_list[0][1] + 1]
        # for i in f_back_list:   # 遍历拐点序列，以拐点为圆心，length为半径来提取相应的速度特征
        # print(f_back_list[0][0])
        v_mean_point_list.append(np.mean(f_back_v_list))
        v_std_point_list.append(np.std(f_back_v_list))

        a_mean_point_list.append(np.mean(f_back_a_list))
        a_std_point_list.append(np.std(f_back_a_list))

        dis_mean_point_list.append(np.mean(f_back_dist_list))
        dis_std_point_list.append(np.std(f_back_dist_list))

        turning_point_loc.append(point_loc_list)
        point_fro_back_index.append(f_back_list)

        # v_mean_point.append(v_mean_point_list)
        # v_std_point.append(v_std_point_list)
        # a_mean_point.append(a_mean_point_list)
        # a_std_point.append(a_std_point_list)
        # dist_mean_point.append(dis_mean_point_list)
        # dist_std_point.append(dis_std_point_list)
        # true_angle.append(true_angle_list)
        # point_angle.append(point_angle_list)
        # abs_angle.append(abs(true_angle_list[0] - point_angle_list[0]))
        # print(',,,,,,')

        abs_angle.append(abs(angle - x[point_index[0]]))

    data['turning_point'] = turning_point
    data['loc_point'] = turning_point_loc
    data['f_back_index'] = point_fro_back_index

    dfGroup['true_angle'] = true_angle_list
    dfGroup['point_angle'] = point_angle_list
    dfGroup['turning_point_index'] = turning_point_index
    dfGroup['clock_direction'] = clockwise_direction
    dfGroup['v_mean_point'] = v_mean_point_list
    dfGroup['v_std_point'] = v_std_point_list
    dfGroup['a_mean_point'] = a_mean_point_list
    dfGroup['a_std_point'] = a_std_point_list
    dfGroup['dist_mean_point'] = dis_mean_point_list
    dfGroup['dist_std_point'] = dis_std_point_list
    dfGroup['absangle'] = abs_angle
    # dfGroup['isangle'] = dfGroup['absangle']<40

    # print(dfGroup['isangle'])
    # data['mean_angle'] = mean_angle
    # data['point_angle'] = point_angle
    # print(dfGroup['isangle'])
    # print(dfGroup['true_angle'])
    # print(dfGroup['point_angle'])
    # print(dfGroup)

    return data, dfGroup


def get_turning_point3(data):  # 暂时以该方式进行拐点特征提取，后期进行改进
    # 获取轨迹的拐点位置列表
    dfGroup = pd.DataFrame()
    turning_point = []  # 记录拐点位置的序列
    point_angle = []  # 记录拐点的偏转角（手绘满足要求的偏转角），记录两个点时使用
    true_angle = []  # 记录拐点的实际夹角，记录两个点时使用
    turning_point_loc = []  # 记录拐点的x,y值（位置）
    turning_point_index = []  # 记录拐点的顺序值（若只有一个拐点取，若两个拐点，取其序号值）
    point_fro_back_index = []  # 记录拐点前后满足要求的点的序列（固定半径作为范围）
    abs_angle = []
    true_angle_list = []  # 记录单个点时使用
    point_angle_list = []
    # 获取相应序列范围内的特征，包括速度，加速度，间隔时间等
    v_mean_point_list = []  # 记录拐点前后满足要求的速度均值
    v_std_point_list = []  # 记录拐点前后满足要求的速度标准差
    a_mean_point_list = []  # 记录拐点前后满足要求的加速度均值
    a_std_point_list = []
    dis_mean_point_list = []  # 记录拐点前后满足要求的间隔距离均值
    dis_std_point_list = []
    for x, v, a, t, dist, n, point_x, point_y in zip(data['Angle'], data['v'], data['abs_a'], data['diff_t'],
                                                     data['dist'],
                                                     data['turning_point_dis'], data['x'], data['y']):
        # 获取拐点的位置
        point_index = []
        x_list = sorted(x)
        point_index_value_0 = find_list_index_1(x, x_list[0])[0]
        point_index_value_1 = find_list_index_1(x, x_list[1])[0]
        point_index_value_2 = find_list_index_1(x, x_list[2])[0]
        point_index_value_3 = find_list_index_1(x, x_list[3])[0]
        point_index_value_4 = find_list_index_1(x, x_list[4])[0]

        five_point_list = [point_index_value_0, point_index_value_1, point_index_value_2, point_index_value_3,
                           point_index_value_4]
        point_list = []
        # 首先判断拐点是否在轨迹的首尾处，以首尾3个数区隔开,需要偏移点在中心位置，接着再判断每个候选点是否与角度平均值相差在12°之内
        for i in five_point_list:
            if i > 2 and (len(x) - i) > 3 and abs(np.mean(x) - x[i]) > 15:
                point_list.append(i)
        if len(point_list) > 0:
            point_index.append(point_list[0])  # 将满足要求的转角弧度最大的点添加到列表
        else:
            point_index.append(n[0])
        turning_point.append(sorted(point_index))

        # 获取拐点前后相应范围内的满足要求的点的序列
        # true_angle_list = []
        point_loc_list = []
        f_back_list = []
        point_index = sorted(point_index)  # 将拐点序列列表排序
        for i in point_index:  # 遍历拐点序列，将对应位置的轨迹偏移角度取出
            point_angle_list.append(x[i])

        for i in point_index:  # 遍历拐点序列，以拐点为圆心，半径length来提取相应的特征
            # 首先获取拐点相应半径的起始点序号和结束序号，以便后面获取速度，加速度等特征
            i_num = 4  # 以首尾5个点作为判断，来计算出拐点的实际坐标位置
            length = 40  # 计算以拐点为圆心，半径为length的范围内的特征值
            if i <= i_num:
                point_1_x, point_1_y = point_x[0], point_y[0]
                point_2_x, point_2_y = point_x[i + i_num], point_y[i + i_num]
                dis_list = []
                for j in range(0, i + i_num):
                    dis = get_ptoline_distance(point_1_x, point_1_y, point_2_x, point_2_y, point_x[j], point_y[j])
                    dis_list.append(dis)
                p_ind = dis_list.index(max(dis_list))
                p_ind_x = point_x[0: i + i_num][p_ind]
                p_ind_y = point_y[0: i + i_num][p_ind]
                k1, k2 = get_radius_point(point_x, point_y, p_ind_x, p_ind_y, length)
                angle = calcul_true_angle(p_ind_x, p_ind_y, point_x[k2], point_y[k2], point_x[k1], point_y[k1])
                true_angle_list.append(angle)
                point_loc_list.append([p_ind_x, p_ind_y])
                f_back_list.append([k1, k2])
            elif len(point_x) - i <= i_num:
                point_1_x, point_1_y = point_x[-1], point_y[-1]
                point_2_x, point_2_y = point_x[i - i_num], point_y[i - i_num]
                dis_list = []
                for j in range(i - i_num, len(point_x)):
                    dis = get_ptoline_distance(point_1_x, point_1_y, point_2_x, point_2_y, point_x[j], point_y[j])
                    dis_list.append(dis)
                p_ind = dis_list.index(max(dis_list))
                p_ind_x = point_x[i - i_num: len(point_x)][p_ind]
                p_ind_y = point_y[i - i_num: len(point_x)][p_ind]
                k1, k2 = get_radius_point(point_x, point_y, p_ind_x, p_ind_y, length)
                angle = calcul_true_angle(p_ind_x, p_ind_y, point_x[k2], point_y[k2], point_x[k1], point_y[k1])
                # angle = calcul_true_angle(point_1_x, point_1_y, point_2_x, point_2_y, p_ind_x, p_ind_y)
                true_angle_list.append(angle)
                point_loc_list.append([p_ind_x, p_ind_y])
                f_back_list.append([k1, k2])
            else:
                point_1_x, point_1_y = point_x[i - i_num], point_y[i - i_num]
                point_2_x, point_2_y = point_x[i + i_num], point_y[i + i_num]
                dis_list = []
                for j in range(i - i_num, i + i_num + 1):
                    dis = get_ptoline_distance(point_1_x, point_1_y, point_2_x, point_2_y, point_x[j], point_y[j])
                    dis_list.append(dis)
                p_ind = dis_list.index(max(dis_list))
                p_ind_x = point_x[i - i_num: i + i_num][p_ind]
                p_ind_y = point_y[i - i_num: i + i_num][p_ind]

                k1, k2 = get_radius_point(point_x, point_y, p_ind_x, p_ind_y, length)
                angle = calcul_true_angle(p_ind_x, p_ind_y, point_x[k2], point_y[k2], point_x[k1], point_y[k1])
                true_angle_list.append(angle)
                point_loc_list.append([p_ind_x, p_ind_y])
                f_back_list.append([k1, k2])

        # #获取相应序列范围内的特征，包括速度，加速度，间隔时间等

        f_back_v_list = v[f_back_list[0][0]:f_back_list[0][1] + 1]
        f_back_a_list = a[f_back_list[0][0]:f_back_list[0][1] + 1]
        # f_back_t_list = t[f_back_list[0][0]:f_back_list[0][1]+1]
        f_back_dist_list = dist[f_back_list[0][0]:f_back_list[0][1] + 1]
        # for i in f_back_list:   # 遍历拐点序列，以拐点为圆心，length为半径来提取相应的速度特征
        # print(f_back_list[0][0])
        v_mean_point_list.append(np.mean(f_back_v_list))
        v_std_point_list.append(np.std(f_back_v_list))

        a_mean_point_list.append(np.mean(f_back_a_list))
        a_std_point_list.append(np.std(f_back_a_list))

        dis_mean_point_list.append(np.mean(f_back_dist_list))
        dis_std_point_list.append(np.std(f_back_dist_list))

        turning_point_loc.append(point_loc_list)
        point_fro_back_index.append(f_back_list)

        abs_angle.append(abs(angle - x[point_index[0]]))

    data['turning_point'] = turning_point
    data['loc_point'] = turning_point_loc
    data['f_back_index'] = point_fro_back_index

    dfGroup['true_angle'] = true_angle_list
    dfGroup['point_angle'] = point_angle_list
    dfGroup['v_mean_point'] = v_mean_point_list
    dfGroup['v_std_point'] = v_std_point_list
    dfGroup['a_mean_point'] = a_mean_point_list
    dfGroup['a_std_point'] = a_std_point_list
    dfGroup['dist_mean_point'] = dis_mean_point_list
    dfGroup['dist_std_point'] = dis_std_point_list
    dfGroup['absangle'] = abs_angle

    return data, dfGroup


# 获取轨迹在每条折线段上的变化数据（以拐点来作为这线段划分标准）#暂时只以弧度变化最大的拐点作为划分点
def get_line_feature(data):
    # 获取轨迹的拐点位置列表
    dfGroup = pd.DataFrame()
    turning_point_v_list = []
    # v_list = []
    t_real_list = []
    t_stay_list = []
    mins_list = []

    for x, y, z, u, n in zip(data['v'], data['a'], data['dist'], data['diff_t'], data['turning_point']):
        v_mean_list = []
        dist_mean_list = []

        t_real_list.append(int(np.sum(z) / np.mean(x)))  # 除去停留时间后，真正的轨迹运行时间
        t_stay_list.append(abs(np.sum(u) - int(np.sum(z) / np.mean(x))))  # 轨迹的原始总时间。两者相减为轨迹在拐点处的停留时间。停留时间可以与拐点的

        mins_list.append(int(abs(np.sum(z) / np.mean(x) / 18 - len(z))))

        if len(n) > 1:
            v_mean_1 = np.mean(x[:n[0] + 1])
            v_mean_2 = np.mean(x[n[0] + 1: n[1] + 1])
            v_mean_3 = np.mean(x[n[1] + 1:])

            v_std_1 = np.std(x[:n[0] + 1])
            v_std_2 = np.std(x[n[0] + 1: n[1] + 1])
            v_std_3 = np.std(x[n[1] + 1:])

            dist_mean_1 = np.sum(z[:n[0] + 1])
            dist_mean_2 = np.sum(z[n[0] + 1: n[1] + 1])
            dist_mean_3 = np.sum(z[n[1] + 1:])

            dist_std_1 = np.std(x[:n[0] + 1])
            dist_std_2 = np.std(x[n[0] + 1: n[1] + 1])
            dist_std_3 = np.std(x[n[1] + 1:])

            dist_max = np.max([dist_std_1, dist_std_2, dist_std_3])
            dist_min = np.min([dist_std_1, dist_std_2, dist_std_3])

            dist_mean_11 = np.mean(z[:n[0] + 1])
            dist_mean_21 = np.mean(z[n[0] + 1: n[1] + 1])
            dist_mean_31 = np.mean(z[n[1] + 1:])

            v_mean_list.append([v_mean_1, v_mean_2, v_mean_3])
            dist_mean_list.append([dist_mean_1, dist_mean_2, dist_mean_3])

    data['t_real'] = t_real_list
    data['t_stay'] = t_stay_list

    return data


# 对距离值进行特征提取
def get_dist_feature(df):
    diff_dist_max_min = get_feature_max_min(df['dist'], 'dist')  # 提取相邻采样位移的最大最小特征，由于最小值一样，用最大值代替峰谷值
    ## 提取相邻采样位移的峰谷值特征，因为相邻位移的最小值差不多，提取峰谷值特征相当于提取相邻位移的最大值特征
    # 前端采样最小距离为3个像素距离，如dist_min小于3，则可以判断为异常（伪造）数据

    ## diff_dist_ptp = get_feature_ptp(df['dist'], 'dist')
    diff_dist_mean_std = get_feature_mean_std(df['dist'], 'dist')  # 提取相邻采样位移的均值和标准差特征
    diff_dist_start_end_mean_std = get_feature_start_end_mean_std(df['dist'], 'dist')  # 获取前后15段位移的均值与方差

    diff_dist_x_abs_max = get_feature_abs_max(df['diff_x'], 'diff_x')  # 比较相邻采样间x轴位移的绝对值的最大值
    diff_dist_x_mean_std = get_feature_mean_std(df['abs_diff_x'], 'diff_x')  # 提取相邻采样位移x轴向的均值和标准差特征

    # diff_dist_y_max_min = get_feature_max_min(df['diff_y'], 'diff_y')
    diff_dist_y_abs_max = get_feature_abs_max(df['diff_y'], 'diff_y')  # 比较相邻采样间y轴位移的绝对值的最大值
    diff_dist_y_mean_std = get_feature_mean_std(df['abs_diff_y'], 'diff_y')  # 提取相邻采样位移y轴向的均值和标准差特征

    dist = pd.concat(
        [diff_dist_max_min, diff_dist_mean_std, diff_dist_start_end_mean_std, diff_dist_x_abs_max, diff_dist_x_mean_std,
         diff_dist_y_abs_max, diff_dist_y_mean_std], axis=1)
    return dist


# 对时间值进行特征提取
def get_time_feature(df):
    # 时间进行特征提取
    t_ptp = get_feature_ptp(df['t'], 't')  # 提取耗时时间特征（时间的最大最小值差）
    t_max_min = get_feature_max_min(df['t'], 't')
    t_start_end_mean_std = get_feature_start_end_mean_std(df['t'], 't')  # 获取前后15个时间段内的均值与方差
    t_diff_max_min = get_feature_max_min(df['diff_t'], 'diff_t')  # 根据最大值除以最小值来计算在拐角时的点数
    t_diff_mean_std = get_feature_mean_std(df['diff_t'], 'diff_t')  # 提取间隔时间的均值和标准差值特征
    t_diff_start_end_mean_std = get_feature_start_end_mean_std(df['diff_t'], 'diff_t')  # 获取前后15个时间段内间隔时间的均值与方差

    t = pd.concat([t_ptp, t_max_min, t_start_end_mean_std, t_diff_max_min, t_diff_mean_std, t_diff_start_end_mean_std],
                  axis=1)

    return t


# 对速度值进行特征提取
def get_v_feature(df):
    # 速度进行特征提取，后期如有需要，可以考虑继续提取速度在x,y轴上的分量特征

    v_max_min = get_feature_max_min(df['v'], 'v')  # 提取速度的最大最小特征
    v_ptp = get_feature_ptp(df['v'], 'v')  # 提取速度峰谷特征（速度的最大最小值差）
    v_mean_std = get_feature_mean_std(df['v'], 'v')  # 提取速度的均值和标准差特征
    v_start_end = get_feature_start_end(df['v'], 'v')

    # 间隔速度进行特征提取
    v_diff_abs_max = get_feature_abs_max(df['abs_diff_v'], 'diff_v')

    # 比较相邻采样间速度的绝对值的最大值（只提取速度的大小值，忽略速度的加减方向）
    # v_diff_ptp = get_feature_ptp(df['diff_v'], 'diff_v')
    v_diff_ptp = get_feature_ptp(df['abs_diff_v'], 'diff_v')
    v_diff_mean_std = get_feature_mean_std(df['abs_diff_v'], 'diff_v')

    v = pd.concat([v_max_min, v_ptp, v_mean_std, v_start_end, v_diff_abs_max, v_diff_ptp, v_diff_mean_std], axis=1)

    return v


# 对加速度值进行特征提取
def get_a_feature(df):
    # 加速度进行特征提取，后期如有需要，可以考虑继续提取加速度在x,y轴上的分量特征
    # a_max_min = get_feature_max_min(df['a'], 'a')  # 提取加速度的最大最小特征，现以加速度绝对值代替。
    a_abs_max = get_feature_abs_max(df['abs_a'], 'a')  # 提取加速度的绝对值的最大值（忽略加速度的方向）
    a_ptp = get_feature_ptp(df['abs_a'], 'a')  # 提取加速度峰谷特征（速度的最大最小值差）
    a_mean_std = get_feature_mean_std(df['abs_a'], 'a')  # 提取加速度的均值和标准差特征

    a = pd.concat([a_abs_max, a_ptp, a_mean_std], axis=1)

    return a


# 其他方面的一些特征值提取：采样点的个数，x轴来回滑动（坐标重复）次数，滑动过程中x轴上插值为0的个数，滑动过程中值为0的个数
# 后期补充特征：采样前后20段时间内的特征；位置，距离，时间的熵；位移的连续性得分；
def get_other_feature(data):
    dfGroup = pd.DataFrame()
    dfGroup['point_count'] = data['x'].apply(lambda x: len(x))
    dfGroup['back_num'] = data['diff_x'].apply(get_back_num) + data['diff_y'].apply(get_back_num)
    dfGroup['dist_all'] = data['dist'].apply(lambda x: np.array(x).sum())  # 求取线段的总长
    dfGroup['a_start_end_compare_zero'] = data['a'].apply(
        lambda x: 1 if np.mean(x[:1]) > 0 and np.mean(x[len(x) - 1:]) < 0 else 0)
    # 获取加速度的前2个值均值，然后与0比较，小于0取值为1（正常画之前速度会从0开始增加，则加速度为正值。结束时，速度会变慢，则加速度为负值）
    return dfGroup


def make_df2(df):
    df = data_process(df)
    df = data_diff(df, ['x', 'y', 't'])
    df = abs_data_diff(df, ['diff_x', 'diff_y'])
    df = get_dist(df)
    df = get_v(df)
    df = data_diff(df, ['v', 'v_x', 'v_y'])
    df = abs_data_diff(df, ['diff_v', 'diff_v_x', 'diff_v_y'])
    df = get_a(df)
    df = get_angle(df)

    df = get_turning_change(df)

    point = get_point_feature(df[['x', 'y', 'target_x', 'target_y']])
    dist = get_dist_feature(df[['diff_x', 'diff_y', 'dist_target', 'dist', 'abs_diff_x', 'abs_diff_y']])
    t = get_time_feature(df[['t', 'diff_t']])
    v = get_v_feature(df[['v', 'diff_v', 'abs_diff_v']])
    a = get_a_feature(df[['abs_a']])
    other = get_other_feature(df)

    # df1 = pd.concat([ df,df_, point, dist, t, v, a, other], axis=1)      #后期df考虑去掉，不需要该特征
    df1 = pd.concat([point, dist, t, v, a, other], axis=1)  # 后期df考虑去掉，不需要该特征

    return df1.fillna(method='bfill')


def make_df(df):
    # df = data_process(df)
    df = data_diff(df, ['x', 'y', 't'])
    df = abs_data_diff(df, ['diff_x', 'diff_y'])
    df = get_dist(df)
    df = get_v(df)
    df = data_diff(df, ['v', 'v_x', 'v_y'])
    df = abs_data_diff(df, ['diff_v', 'diff_v_x', 'diff_v_y'])
    df = get_a(df)
    df = get_angle(df)

    # df = get_turning_point1(df)
    # df,df_ = get_turning_point(df)
    # df = get_line_feature(df)
    # df = abs_data_diff(df, ['x', 'y', 't'])

    # df_.to_csv('new_df_6w.csv',index=False)

    # df = get_mean_point(df)
    # df = get_mean_point_angle(df)
    df = get_turning_change(df)
    # df2 = get_v_change(df)

    point = get_point_feature(df[['x', 'y', 'target_x', 'target_y']])
    dist = get_dist_feature(df[['diff_x', 'diff_y', 'dist_target', 'dist', 'abs_diff_x', 'abs_diff_y']])
    t = get_time_feature(df[['t', 'diff_t']])
    v = get_v_feature(df[['v', 'diff_v', 'abs_diff_v']])
    a = get_a_feature(df[['abs_a']])
    other = get_other_feature(df)

    # df1 = pd.concat([ df,df_, point, dist, t, v, a, other], axis=1)      #后期df考虑去掉，不需要该特征
    df1 = pd.concat([point, dist, t, v, a, other], axis=1)  # 后期df考虑去掉，不需要该特征

    return df1.fillna(method='bfill')


def input_df():
    path = r'D:\machine_learning\tensorflow\tfen10_27\Recognition_model\master3'
    file_path = os.path.join(path, 'curve.csv')

    data = pd.read_csv(file_path, names=['id', 'point'], skiprows=2).ix[:]
    id_data = data['id'].copy()
    data.drop('id', axis=1, inplace=True)

    return data, id_data


def train_df():
    # set path
    path = r'D:\machine_learning\tensorflow\tfen10_27\Recognition_model\master'
    # file_path = os.path.join(path, 'test_data_trasnform.csv')
    file_path = os.path.join(path, 'All_data_20.csv')

    data = pd.read_csv(file_path, names=['id', 'point'], skiprows=2).ix[:]
    id_data = data['id'].copy()
    data.drop('id', axis=1, inplace=True)

    return data, id_data


def save_df(df, name, id_data):
    path = r'D:\machine_learning\tensorflow\tfen10_27\Recognition_model\master'
    df.insert(0, 'id', id_data)  # 在行首添加ID列
    train = df.ix[:, :]
    train.to_csv(path + "\\" + name + "train.csv", index=None)


# 计算熵的公式，后期考虑作为添加特征进行处理
def calculate_entropy(data):
    data_space = np.linspace(min(data), max(data), 10)
    cats = pd.cut(data, data_space)
    value_count = pd.value_counts(cats)
    value_count_ = []
    for j in range(len(value_count)):
        value_count_.append(value_count[j])


def cal_score(data, mean, std, f_down, f_up):  # 针对每个特征，进行统计打分，若分数低于一定值，则判定为异常轨迹
    score_1 = 0
    score_2 = 0
    score = 0

    if data >= mean:
        if data - mean > 3 * std:
            score_1 += 2  # 每个特征值评判分数为三分。后期需根据特征重要性，进行权重分配
    else:
        if mean - data > 2 * std:
            score_1 += 2  # 每个特征值评判分数为三分。后期需根据特征重要性，进行权重分配
    if data <= f_down - 1.5 * (f_up - f_down) or data >= f_up + 1.5 * (f_up - f_down):  # 以上下四分位数和四分位距来进行判断是否为异常值
        score_2 += 2

    if score_1 > 0 or score_2 > 0:
        score += 2

    return score


def cal_score1(data, mean, std, down):  # 针对每个特征，进行统计打分，若分数低于一定值，则判定为异常轨迹
    score_1 = 0
    score_2 = 0
    score = 0
    if data >= mean:
        if data - mean > 3.8 * std:
            score_1 += 2  # 每个特征值评判分数为三分。后期需根据特征重要性，进行权重分配
    else:
        if mean - data > 2 * std or data < down:
            score_1 += 2  # 每个特征值评判分数为三分。后期需根据特征重要性，进行权重分配

    return score_1


def re_train(df):  # 针对原始数据，进行统计打分，若分数低于一定值，则判定为异常轨迹，并进行标记

    # 对每个输入的轨迹点进行检测
    a = pd.read_csv('mean_std_data_all.csv')
    # dfGroup = pd.DataFrame()
    score_list = []
    for j in range(len(df)):
        score = 0
        for i in range(len(df.columns)):
            mean_value = a.ix[i][1]
            std_value = a.ix[i][2]
            # down_value =a.ix[i][3]
            # up_value =a.ix[i][4]
            score += cal_score1(df.ix[j][i], mean_value, std_value)
            # score += cal_score(df.ix[j][i], mean_value, std_value, down_value, up_value)
        score_list.append(score)
    df['pred'] = score_list
    df['isAnomaly'] = df['pred'] < 10
    # df.to_csv('pred_anoma_labe_result.csv')

    return df


def get_feature_mean_std_data(df):
    # 将传入的数据进行特征抽取，形成对比模型数据
    a = df.mean(axis=0)
    b = df.std(axis=0)
    df_down_list = []
    df_up_list = []
    for i in range(len(df.columns)):
        z = df.iloc[:, i]
        df_25 = np.percentile(z, 25)
        df_75 = np.percentile(z, 75)
        df_down_list.append(df_25)
        df_up_list.append(df_75)
    c = pd.concat([a, b], axis=1)
    c.columns = ['mean', 'std']
    df_down_list_ = pd.DataFrame(df_down_list, columns=['down25'], index=c.index)
    df_up_list_ = pd.DataFrame(df_up_list, columns=['up75'], index=c.index)
    cd = pd.concat([a, b, df_down_list_, df_up_list_], axis=1)

    return cd


def judge_is_ai(data):  # 针对输入进去的每个轨迹点，与模型进行对比，若返回值都为1，则说明正常模型，否则错误

    # 对每个输入的轨迹点进行转换
    # data = eval(data)
    data = transform_dic(data)
    data = np.array(data)
    data = pd.DataFrame({"point": str(data)}, index=['id'])
    data['point'] = data['point'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[1:]])
    # 提取 x坐标 y坐标 t 目标点x坐标  目标点y坐标（起始点作为目标点）
    df = pd.DataFrame()
    df['x'] = data['point'].apply(lambda x: np.array(x)[:, 0])
    df['y'] = data['point'].apply(lambda x: np.array(x)[:, 1])
    df['t'] = data['point'].apply(lambda x: np.array(x)[:, 2])
    df['target_x'] = data['point'].apply(lambda x: np.array(x)[0, 0])
    df['target_y'] = data['point'].apply(lambda x: np.array(x)[0, 1])

    data = make_df(df)
    data = data.fillna(0)

    # 应用模型进行预测
    X_cols = ['x_ptp', 'y_ptp', 'dist_max', 'dist_min', 'dist_mean', 'dist_std', 'dist_start_mean', 'dist_end_mean',
              'dist_start_std', 'dist_end_std', 'diff_x_abs_max', 'diff_x_mean', 'diff_x_std', 'diff_y_abs_max',
              'diff_y_mean', 'diff_y_std', 't_ptp', 't_max', 't_min', 'diff_t_max', 'diff_t_min', 'v_max', 'v_min',
              'v_ptp', 'v_mean', 'v_std', 'v_start', 'v_end', 'diff_v_abs_max', 'diff_v_ptp', 'diff_v_mean',
              'diff_v_std', 'a_abs_max', 'a_ptp', 'a_mean', 'a_std']

    Y_cols = ['point_count', 'back_num', 'dist_all']
    # result1 = ILF1.predict(data[X_cols][:])
    # result2 = ILF2.predict(data[Y_cols][:])

    # return result1[0], result2[0]


if __name__ == '__main__':
    A1 = [{"x": 74, "y": 156, "time": 0}, {"x": 81, "y": 152, "time": 59}, {"x": 102, "y": 144, "time": 77},
          {"x": 137, "y": 130, "time": 92}, {"x": 210, "y": 101, "time": 110}, {"x": 260, "y": 87, "time": 126},
          {"x": 309, "y": 74, "time": 143}, {"x": 345, "y": 64, "time": 160}, {"x": 366, "y": 60, "time": 176},
          {"x": 375, "y": 60, "time": 193}, {"x": 378, "y": 61, "time": 226}, {"x": 378, "y": 66, "time": 243},
          {"x": 370, "y": 80, "time": 259}, {"x": 342, "y": 107, "time": 276}, {"x": 259, "y": 162, "time": 293},
          {"x": 207, "y": 194, "time": 309}, {"x": 164, "y": 218, "time": 326}, {"x": 147, "y": 229, "time": 343}]

    test_list = [{"x": 106, "y": 139, "time": 0}, {"x": 106, "y": 130, "time": 8}, {"x": 106, "y": 113, "time": 25},
                 {"x": 112, "y": 89, "time": 42}, {"x": 118, "y": 75, "time": 59}, {"x": 126, "y": 65, "time": 76},
                 {"x": 34, "y": 58, "time": 92}, {"x": 140, "y": 55, "time": 109}, {"x": 146, "y": 59, "time": 142},
                 {"x": 156, "y": 70, "time": 159}, {"x": 179, "y": 94, "time": 175}, {"x": 207, "y": 127, "time": 192},
                 {"x": 243, "y": 165, "time": 209}, {"x": 274, "y": 195, "time": 225},
                 {"x": 308, "y": 226, "time": 242}, {"x": 320, "y": 239, "time": 259},
                 {"x": 326, "y": 246, "time": 275}, {"x": 328, "y": 249, "time": 292}]
    # (v1, v2) = judge_is_ai(test_list)
    # print(judge_is_ai(test_list))
    test_list1 = transform_dic(test_list)
    data = np.array(test_list1)
    data = pd.DataFrame({"point": str(data)}, index=['id'])
    data['point'] = data['point'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[1:]])
    # 提取 x坐标 y坐标 t 目标点x坐标  目标点y坐标（起始点作为目标点）
    df = pd.DataFrame()

    df['x'] = data['point'].apply(lambda x: np.array(x)[:, 0])
    df['y'] = data['point'].apply(lambda x: np.array(x)[:, 1])
    df['t'] = data['point'].apply(lambda x: np.array(x)[:, 2])
    df['target_x'] = data['point'].apply(lambda x: np.array(x)[0, 0])
    df['target_y'] = data['point'].apply(lambda x: np.array(x)[0, 1])
    df = data_diff(df, ['x', 'y', 't'])
    df = abs_data_diff(df, ['diff_x', 'diff_y'])
    df = get_dist(df)
    df = get_v(df)
    df = data_diff(df, ['v', 'v_x', 'v_y'])
    df = abs_data_diff(df, ['diff_v', 'diff_v_x', 'diff_v_y'])
    df = get_a(df)
    num0 = 0
    num1 = len(test_list)

    print(df['dist'][0][num0:num1])
    print(df['diff_t'][0][num0:num1])
    print(df['v'][0][num0:num1])
