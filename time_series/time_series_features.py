#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   time_series_features.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/22 19:30   SeafyLiang   1.0      时间特征构造以及时间序列特征构造
"""
'''
参考资料：https://mp.weixin.qq.com/s/CWUFLMK0ZhDuqWXiveoBpg

总结
1.时间特征主要有两大类：

1）从时间变量提取出来的特征
如果每条数据为一条训练样本，时间变量提取出来的特征可以直接作为训练样本的特征使用。
例子：用户注册时间变量。对于每个用户来说只有一条记录，提取出来的特征可以直接作为训练样本的特征使用，不需要进行二次加工。
如果每条数据不是一条训练样本，时间变量提取出来的特征需要进行二次加工（聚合操作）才能作为训练样本的特征使用。
例子：用户交易流水数据中的交易时间。由于每个用户的交易流水数量不一样，从而导致交易时间提取出来的特征的数据不一致，所以这些特征不能直接作为训练样本的特征来使用。我们需要进一步进行聚合操作才能使用，如先从交易时间提取出交易小时数，然后再统计每个用户在每个小时（1-24小时）的交易次数来作为最终输出的特征。
2）对时间变量进行条件过滤，然后再对其他变量进行聚合操作所产生的特征
主要是针对类似交易流水这样的数据，从用户角度进行建模时，每个用户都有不定数量的数据，因此需要对数据进行聚合操作来为每个用户构造训练特征。而包含时间的数据，可以先使用时间进行条件过滤，过滤后再构造聚合特征。
2. 时间序列数据可以从带有时间的流水数据统计得到，实际应用中可以分别从带有时间的流水数据以及时间序列数据中构造特征，这些特征可以同时作为模型输入特征。
例如：美团的商家销售量预测中，每个商家的交易流水经过加工后可以得到每个商家每天的销售量，这个就是时间序列数据。

'''
import pandas as pd

# 1、时间特征构造
# 1.1 连续值时间特征与离散值时间特征
'''
1、连续值时间特征
● 持续时间（单页浏览时长）；
● 间隔时间；
  ○ 上次购买/点击离现在的时长；
  ○ 产品上线到现在经过的时长；
2、离散型时间特征
1）时间特征拆解
年；月；日；时；分；秒；
一天中的第几分钟；星期几；
一年中的第几天；一年中的第几个周；一年中的哪个季度；
一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
'''
# 构造时间数据
date_time_str_list = [
    '2019-01-01 01:22:26', '2019-02-02 04:34:52', '2019-03-03 06:16:40',
    '2019-04-04 08:11:38', '2019-05-05 10:52:39', '2019-06-06 12:06:25',
    '2019-07-07 14:05:25', '2019-08-08 16:51:33', '2019-09-09 18:28:28',
    '2019-10-10 20:55:12', '2019-11-11 22:55:12', '2019-12-12 00:55:12',
]
df = pd.DataFrame({'时间': date_time_str_list})
# 把字符串格式的时间转换成Timestamp格式
df['时间'] = df['时间'].apply(lambda x: pd.Timestamp(x))

# 年份
df['年'] = df['时间'].apply(lambda x: x.year)

# 月份
df['月'] = df['时间'].apply(lambda x: x.month)

# 日
df['日'] = df['时间'].apply(lambda x: x.day)

# 小时
df['时'] = df['时间'].apply(lambda x: x.hour)

# 分钟
df['分'] = df['时间'].apply(lambda x: x.minute)

# 秒数
df['秒'] = df['时间'].apply(lambda x: x.second)

# 一天中的第几分钟
df['一天中的第几分钟'] = df['时间'].apply(lambda x: x.minute + x.hour * 60)

# 星期几；
df['星期几'] = df['时间'].apply(lambda x: x.dayofweek)

# 一年中的第几天
df['一年中的第几天'] = df['时间'].apply(lambda x: x.dayofyear)

# 一年中的第几周
df['一年中的第几周'] = df['时间'].apply(lambda x: x.week)

# 一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
period_dict = {
    23: '深夜', 0: '深夜', 1: '深夜',
    2: '凌晨', 3: '凌晨', 4: '凌晨',
    5: '早晨', 6: '早晨', 7: '早晨',
    8: '上午', 9: '上午', 10: '上午', 11: '上午',
    12: '中午', 13: '中午',
    14: '下午', 15: '下午', 16: '下午', 17: '下午',
    18: '傍晚',
    19: '晚上', 20: '晚上', 21: '晚上', 22: '晚上',
}
df['时间段'] = df['时'].map(period_dict)

# 一年中的哪个季度
season_dict = {
    1: '春季', 2: '春季', 3: '春季',
    4: '夏季', 5: '夏季', 6: '夏季',
    7: '秋季', 8: '秋季', 9: '秋季',
    10: '冬季', 11: '冬季', 12: '冬季',
}
df['季节'] = df['月'].map(season_dict)
print(df.head(5))

# 1.2 2）时间特征判断
# 是否闰年；是否月初；是否月末；
# 是否季节初；是否季节末；
# 是否年初；是否年尾；
# 是否周末；是否公共假期；是否营业时间；
# 两个时间间隔之间是否包含节假日/特殊日期；
# 构造时间数据
date_time_str_list = [
    '2010-01-01 01:22:26', '2011-02-03 04:34:52', '2012-03-05 06:16:40',
    '2013-04-07 08:11:38', '2014-05-09 10:52:39', '2015-06-11 12:06:25',
    '2016-07-13 14:05:25', '2017-08-15 16:51:33', '2018-09-17 18:28:28',
    '2019-10-07 20:55:12', '2020-11-23 22:55:12', '2021-12-25 00:55:12',
    '2022-12-27 02:55:12', '2023-12-29 03:55:12', '2024-12-31 05:55:12',
]
df = pd.DataFrame({'时间': date_time_str_list})
# 把字符串格式的时间转换成Timestamp格式
df['时间'] = df['时间'].apply(lambda x: pd.Timestamp(x))

# 是否闰年
df['是否闰年'] = df['时间'].apply(lambda x: x.is_leap_year)

# 是否月初
df['是否月初'] = df['时间'].apply(lambda x: x.is_month_start)

# 是否月末
df['是否月末'] = df['时间'].apply(lambda x: x.is_month_end)

# 是否季节初
df['是否季节初'] = df['时间'].apply(lambda x: x.is_quarter_start)

# 是否季节末
df['是否季节末'] = df['时间'].apply(lambda x: x.is_quarter_end)

# 是否年初
df['是否年初'] = df['时间'].apply(lambda x: x.is_year_start)

# 是否年尾
df['是否年尾'] = df['时间'].apply(lambda x: x.is_year_end)

# 是否周末
df['是否周末'] = df['时间'].apply(lambda x: True if x.dayofweek in [5, 6] else False)

# 是否公共假期
public_vacation_list = [
    '20190101', '20190102', '20190204', '20190205', '20190206',
    '20190207', '20190208', '20190209', '20190210', '20190405',
    '20190406', '20190407', '20190501', '20190502', '20190503',
    '20190504', '20190607', '20190608', '20190609', '20190913',
    '20190914', '20190915', '20191001', '20191002', '20191003',
    '20191004', '20191005', '20191006', '20191007',
]  # 此处未罗列所有公共假期
df['日期'] = df['时间'].apply(lambda x: x.strftime('%Y%m%d'))
df['是否公共假期'] = df['日期'].apply(lambda x: True if x in public_vacation_list else False)

# 是否营业时间
df['是否营业时间'] = False
df['小时'] = df['时间'].apply(lambda x: x.hour)
df.loc[((df['小时'] >= 8) & (df['小时'] < 22)), '是否营业时间'] = True

df.drop(['日期', '小时'], axis=1, inplace=True)
print(df.head(5))


# 2、时间序列特征构造
'''
时间序列不仅包含一维时间变量，还有一维其他变量，如股票价格、天气温度、降雨量、订单量等。时间序列分析的主要目的是基于历史数据来预测未来信息。对于时间序列，我们关心的是长期的变动趋势、周期性的变动（如季节性变动）以及不规则的变动。
按固定时间长度把时间序列划分成多个时间窗，然后构造每个时间窗的特征。
1.时间序列聚合特征
按固定时间长度把时间序列划分成多个时间窗，然后使用聚合操作构造每个时间窗的特征。
1）平均值
例子：历史销售量平均值、最近N天销售量平均值。
2）最小值
例子：历史销售量最小值、最近N天销售量最小值。
3）最大值
例子：历史销售量最大值、最近N天销售量最大值。
4）扩散值
分布的扩散性，如标准差、平均绝对偏差或四分位差，可以反映测量的整体变化趋势。
5）离散系数值
离散系数是策略数据离散程度的相对统计量，主要用于比较不同样本数据的离散程度。
6）分布性
时间序列测量的边缘分布的高阶特效估计(如偏态系数或峰态系数)，或者更进一步对命名分布进行统计测试(如标准或统一性)，在某些情况下比较有预测力。
'''
# 加载洗发水销售数据集
df = pd.read_csv('data/shampoo-sales.csv')
df.dropna(inplace=True)
df.rename(columns={'Sales of shampoo over a three year period': 'value'}, inplace=True)

# 平均值
mean_v = df['value'].mean()
print('mean: {}'.format(mean_v))

# 最小值
min_v = df['value'].min()
print('min: {}'.format(min_v))

# 最大值
max_v = df['value'].max()
print('max: {}'.format(max_v))

# 扩散值：标准差
std_v = df['value'].std()
print('std: {}'.format(std_v))

# 扩散值：平均绝对偏差
mad_v = df['value'].mad()
print('mad: {}'.format(mad_v))

# 扩散值：四分位差
q1 = df['value'].quantile(q=0.25)
q3 = df['value'].quantile(q=0.75)
irq = q3 - q1
print('q1={}, q3={}, irq={}'.format(q1, q3, irq))

# 离散系数
variation_v = std_v/mean_v
print('variation: {}'.format(variation_v))

# 分布性：偏态系数
skew_v = df['value'].skew()
print('skew: {}'.format(skew_v))
# 分布性：峰态系数
kurt_v = df['value'].kurt()
print('kurt: {}'.format(kurt_v))

# 输出：
# mean: 312.59999999999997
# min: 119.3
# max: 682.0
# std: 148.93716412347473
# mad: 119.66666666666667
# q1=192.45000000000002, q3=411.1, irq=218.65
# variation: 0.47644646232717447
# skew: 0.8945388528534595
# kurt: 0.11622821118738624

# 2.2 时间序列历史特征
'''
1）前一（或n）个窗口的取值
例子：昨天、前天和3天前的销售量。
2）周期性时间序列前一（或n）周期的前一（或n）个窗口的取值
例子：写字楼楼下的快餐店的销售量一般具有周期性，周期长度为7天，7天前和14天前的销售量。
'''
# 加载洗发水销售数据集
df = pd.read_csv('data/shampoo-sales.csv')
df.dropna(inplace=True)
df.rename(columns={'Sales of shampoo over a three year period': 'value'}, inplace=True)


df['-1day'] = df['value'].shift(1)
df['-2day'] = df['value'].shift(2)
df['-3day'] = df['value'].shift(3)

df['-1period'] = df['value'].shift(1*12)
df['-2period'] = df['value'].shift(2*12)

print(df.head(60))

# 2.3 3.时间序列复合特征
'''
# 1）趋势特征
# 趋势特征可以刻画时间序列的变化趋势。
# 例子：每个用户每天对某个Item行为次数的时间序列中，User一天对Item的行为次数/User三天对Item的行为次数的均值，表示短期User对Item的热度趋势，大于1表示活跃逐渐在提高；三天User对Item的行为次数的均值/七天User对Item的行为次数的均值表示中期User对Item的活跃度的变化情况；七天User对Item的行为次数的均值/ 两周User对Item的行为次数的均值表示“长期”（相对）User对Item的活跃度的变化情况。
'''
# 加载洗发水销售数据集
df = pd.read_csv('data/shampoo-sales.csv')
df.dropna(inplace=True)
df.rename(columns={'Sales of shampoo over a three year period': 'value'}, inplace=True)

df['last 3 day mean'] = (df['value'].shift(1) + df['value'].shift(2) + df['value'].shift(3))/3
df['最近3天趋势'] = df['value'].shift(1)/df['last 3 day mean']
print(df.head(60))

# 2）窗口差异值特征
# 一个窗口到下一个窗口的差异。例子：商店销售量时间序列中，昨天的销售量与前天销售量的差值。
# 加载洗发水销售数据集
df = pd.read_csv('data/shampoo-sales.csv')
df.dropna(inplace=True)
df.rename(columns={'Sales of shampoo over a three year period': 'value'}, inplace=True)

df['最近两月销量差异值'] = df['value'].shift(1) - df['value'].shift(2)
print(df.head(60))

# 3）自相关性特征
# 原时间序列与自身左移一个时间空格（没有重叠的部分被移除）的时间序列相关联。
import pandas as pd
# 加载洗发水销售数据集
df = pd.read_csv('data/shampoo-sales.csv')
df.dropna(inplace=True)
df.rename(columns={'Sales of shampoo over a three year period': 'value'}, inplace=True)

print('滞后数为1的自相关系数：{}'.format(df['value'].autocorr(1)))
print('滞后数为2的自相关系数：{}'.format(df['value'].autocorr(2)))
# 输出：
# 滞后数为1的自相关系数：0.7194822398024308
# 滞后数为2的自相关系数：0.8507433352850972