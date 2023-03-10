#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   best_ks.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/3/10 16:14   SeafyLiang   1.0      BestKS分箱寻找最近分箱策略
"""
'''
参考资料：https://blog.csdn.net/yeshang_lady/article/details/112623604
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def univeral_df(data, feature, target, total_name, good_name, bad_name):
    """
    统计feature不同值对应的target数量
    feature: 将要进行分割的变量
    target: 目标变量，其值只能取0和1 1为坏样本，0为好样本
    """
    data = data[[feature, target]]
    result = data.groupby(feature)[target].agg(['count', 'sum'])
    result = result.sort_index()
    result.columns = [total_name, bad_name]
    result = result.fillna(0)
    result[good_name] = result[total_name] - result[bad_name]
    result = result.reset_index()
    return result


def get_max_ks(data_df, start, end, rate, total_name, good_name, bad_name):
    """
    寻找能满足的分箱占比的最大的KS对应的值
    """
    total_all = data_df[total_name].sum()
    limit = rate * total_all
    data_cut = data_df.loc[start:end, :]  # data_cut统计的范围包含end
    data_cut['bad_rate_cum'] = (data_cut[bad_name] / data_cut[bad_name].sum()).cumsum()
    data_cut['good_rate_cum'] = (data_cut[good_name] / data_cut[good_name].sum()).cumsum()
    data_cut['total_cum'] = data_cut[total_name].cumsum()
    data_cut['total_other_cum'] = total_all - data_cut[total_name]
    data_cut['KS'] = np.abs(data_cut['bad_rate_cum'] - data_cut['good_rate_cum'])
    try:
        cut = data_cut[(data_cut['total_cum'] >= limit) & (data_cut['total_other_cum'] >= limit)]['KS'].idxmax()
        return cut
    except:
        return np.nan


def verify_cut(data_df, cut_list, total_name, good_name, bad_name):
    """
    判断是否能继续分箱下去,返回True(继续进行切割)或False(不继续切割)。具体判断条件如下：
    1 是否存在某箱对应的类别全为0或1
    2 现有的分箱能否保证bad rate递增或递减
    3 woe的单调性和bad rate的单调性相反(这个条件感觉太严格了)
    """
    bad_all = data_df[bad_name].sum()
    good_all = data_df[good_name].sum()
    cut_bad = np.array([data_df.loc[x[0]:x[1], bad_name].sum() for x in cut_list])
    cut_good = np.array([data_df.loc[x[0]:x[1], good_name].sum() for x in cut_list])
    cond1 = (0 not in cut_bad) and (0 not in cut_good)
    cut_bad_rate = cut_bad / bad_all
    cut_good_rate = cut_good / good_all
    cut_woe = np.log(cut_bad_rate / cut_good_rate)
    cond2 = sorted(cut_woe, reverse=False) == list(cut_woe) and sorted(cut_bad_rate, reverse=True) == list(cut_bad_rate)
    cond3 = sorted(cut_woe, reverse=True) == list(cut_woe) and sorted(cut_bad_rate, reverse=False) == list(cut_bad_rate)
    cond4 = sorted(cut_bad_rate, reverse=False) == list(cut_bad_rate) or sorted(cut_bad_rate, reverse=True) == list(
        cut_bad_rate)
    return cond1 and cond4


def cut_fun(data_df, start, end, rate, total_name, good_name, bad_name):
    """
    对从start到end这一段数据进行下一步切分，并返回新的割点对
    """
    cut = get_max_ks(data_df, start, end, rate, total_name, good_name, bad_name)
    if cut:
        return [(start, cut), (cut + 1, end)]
    else:
        return [(start, end)]


def ks_cut_main(data_df, feature, rate, total_name, good_name, bad_name, bins=None, null_value=False,
                missing_value=[]):
    """
    bins: 分箱数。默认为None,即不限定分箱数目。若为int,则为指定的分箱数
    null_value: 布尔型。字段中填充的缺失值是否需要单独划分为一箱。若为True,则单独划分为一箱。
    missing_value:若null_value为True,则missing_value中的数据即为缺失值,每个缺失值会单独成一箱
    """
    if null_value and missing_value:
        data_df = data_df[~data_df[feature].isin(missing_value)]
    start = data_df.index.min()
    end = data_df.index.max()
    cut_list = [(start, end)]  # 真正有效的割点集合
    tt = cut_list.copy()
    for cut_seg in tt:
        cut_bin = cut_fun(data_df, cut_seg[0], cut_seg[1], rate, total_name, good_name, bad_name)
        if len(cut_bin) > 1:
            temp = cut_list.copy()
            index_seg = temp.index(cut_seg)
            temp[index_seg] = cut_bin[0]
            temp.insert(index_seg + 1, cut_bin[1])
            if verify_cut(data_df, temp, total_name, good_name, bad_name):
                cut_list = temp
                tt.extend(cut_bin)
        if bins and len(cut_list) > bins:  # 判断是否达到限定的分箱数
            break
    # 将割点对转化为对应的数值
    # cut_list中保留的割点对中的数据为data_df中的inde，这里想要获得真正的feature的割点数据则需要依据data_df的index找到对应的feature字段的值
    cut_list = sorted([-np.inf] + [data_df.loc[item[0], feature] for item in cut_list] + [np.inf] + missing_value)
    return cut_list


def ks_best_cut(x_value: np.ndarray, y_value: np.ndarray, bins=None):
    data = pd.DataFrame([x_value, y_value]).T
    data.columns = ['x', 'target']
    uni_result = univeral_df(data, 'x', 'target', 'total', 'good', 'bad')
    cut_bin_result = ks_cut_main(data_df=uni_result, feature='x', rate=0.05, total_name='total',
                                 good_name='good', bad_name='bad', bins=bins)
    return cut_bin_result

def best_ks_test_method():
    data_x, data_y = make_classification(n_samples=1000, n_classes=2, n_features=4, n_informative=2, random_state=0)
    # data_y中：1为坏样本，0为好样本
    df = pd.read_csv('df.csv')
    col_name_list = ['a', 'b']
    for col_name in col_name_list:
        print('best_kes：', col_name)
        ks_cut = ks_best_cut(df[col_name].values, df['fault_label'].values, bins=10)
        # 对A进行分箱
        x_bin_value = pd.cut(df[col_name].values, ks_cut, right=False)
        print(x_bin_value.value_counts())


if __name__ == '__main__':
    best_ks_test_method()