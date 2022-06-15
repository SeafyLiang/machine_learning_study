#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   surpriseDemo.py
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022-03-29 21:12:18   SeafyLiang   1.0       scikit-surprise实现推荐
"""
# 可以使用上面提到的各种推荐系统算法
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

from surprise import accuracy

from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

from surprise import KNNBasic
from surprise import Dataset

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo = KNNBasic()
algo.fit(trainset)

uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

inner_id = algo.trainset.to_inner_iid(
    iid
)
algo.get_neighbors(inner_id, k=10)
algo.trainset.to_raw_iid(108)

import pandas

ratings = pandas.read_csv(
    "u.data",
    sep='\t', names=["UserID", "ItemID", "rating", "timestamp"]
)

from surprise import Reader
from surprise import Dataset

reader = Reader(
    rating_scale=(1, 5)
)

ratingDataSet = Dataset.load_from_df(
    ratings[['UserID', 'ItemID', 'rating']],
    reader
)

from surprise import KNNBasic

# 基于用户的协同推荐算法
userBased = KNNBasic(
    k=40, min_k=3,
    sim_options={'user_based': True}
)
# 从DataSet中调用build_full_trainset方法生成训练样本
trainSet = ratingDataSet.build_full_trainset()
# 使用所有训练样本训练模型
userBased.fit(trainSet)

# 目标用户ID
uid = 196

# 获取 uid 对应的所有 iid
hasItemIDs = ratings[
    ratings.UserID == uid
    ].ItemID.drop_duplicates().values

# 获取所有的 iid
allItemIDs = ratings.ItemID.drop_duplicates()

# 保存 没有的 iid 的预测评分
_iids = []
_ratings = []
for iid in allItemIDs:
    if iid not in hasItemIDs:
        _iids.append(iid)
        # 调用模型的predict方法，预测uid对iid的评分
        _ratings.append(
            userBased.predict(uid, iid).est
        )
    # 将结果以数据框的形式返回
result = pandas.DataFrame({
    'iid': _iids,
    'rating': _ratings
})

itemBased = KNNBasic(
    k=40,
    min_k=3,
    sim_options={'user_based': False}
)
itemBased.fit(
    ratingDataSet.build_full_trainset()
)
# 数据集中的商品ID
iid = 110
# DataSet会对数据集中的物品ID重新进行编码，
# 所以先要找出模型中使用的id，称为inner_id
item_inner_id = itemBased.trainset.to_inner_iid(
    iid
)
# 使用inner_id进行相似商品的计算，找出该商品最接近的10个商品
iid_inner_neighbors = itemBased.get_neighbors(
    item_inner_id, k=10
)
# 把inner_id转换为数据集中的ID
iid_neighbors = [
    itemBased.trainset.to_raw_iid(inner_iid)
    for inner_iid in iid_inner_neighbors
]

items = pandas.read_csv(
    "u.item", sep="\|",
    names=[
        'movie id', 'movie title', 'release date',
        'video release dat', 'IMDb URL', 'unknown',
        'Action', 'Adventure', 'Animation',
        'Children\'s', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ], engine='python'
)
items.index = items['movie id']
