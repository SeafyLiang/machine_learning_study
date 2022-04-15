#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   create_data.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/15 21:07   SeafyLiang   1.0       3种造数据的方法
"""
# 参考资料 https://mp.weixin.qq.com/s/XT9L6TFc80zSZqPsrB1ECg

# 1、Faker
from ctgan import CTGANSynthesizer
from faker import Faker
import warnings

warnings.filterwarnings('ignore')

'''
Python当中的Faker模块主要是用来生成伪数据，包括了城市、姓名等等，并且还支持中文
faker.readthedocs.io/en/master/providers.html
'''
fake = Faker(locale='zh_CN')
# 随机生成一个城市
print(fake.city())
# 随机生成一个地址
print(fake.address())
# 随机生成一个手机号
print(fake.phone_number())

'''
可以通过机器学习算法在基于真实数据的基础上生成合成数据，将后者应用于模型的训练上，例如由MIT的DAI(Data to AI)实验室推出的合成数据开源系统----Synthetic Data Vault(SDV)，该模块可以从真实数据库中构建一个机器学习模型来捕获多个变量之间的相关性，要是原始的数据库中存在着一些缺失值和一些极值，最后在合成的数据集当中也会有一些缺失值与极值。
'''
# 2、sdv
from vega_datasets import data
import pandas as pd

data = data.cars()
data = data.head(10)
print(data.head())
print(data.info())

# 将算法模型拟合到数据集中的数据，我们可以尝试生成一些数据
from sdv.tabular import GaussianCopula

model = GaussianCopula()
model.fit(data)

sample = model.sample(200)
print(sample.head())

# 看一下新生成的数据和真实数据相比相似性几何
from sdv.evaluation import evaluate

print(evaluate(sample, data))
# 0.6507265057782188

# 3、CTGAN
# 对于生成对抗的神经网络GANs而言，其中第一个网络为生成器，而第二个网络为鉴别器，最后生成器产生出来的数据表并没有被鉴别器分辨出其中的差异。
# 此数据未跑通 - -!
discrete_columns = ['Cylinders',
                    'Weight_in_lbs']
ctgan = CTGANSynthesizer(batch_size=50, epochs=5, verbose=False)
ctgan.fit(data, discrete_columns)
# 将训练好的模型保存下来
ctgan.save('ctgan-food-demand.pkl')
# 生成200条数据集
samples = ctgan.sample(200)
print(samples.head())

from sdv.evaluation import evaluate
print(evaluate(samples, data))
