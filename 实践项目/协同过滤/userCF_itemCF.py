#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   userCF_itemCF.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/3/29 21:51   SeafyLiang   1.0     surprise实现UserCF和ItemCF
"""
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise import KNNBasic

# 读取用户评价矩阵
ratings = pd.read_csv('u.data', sep='\t', names=['UserID', 'ItemID', 'rating', 'timestamp'])

# df转DataSet
reader = Reader(rating_scale=(1, 5))
ratingDataSet = Dataset.load_from_df(ratings[['UserID', 'ItemID', 'rating']], reader)

"""
Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set to the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
"""


# 使用knn实现UserCF
def userCF_method():
    userCF = KNNBasic(
        k=40, min_k=3,
        sim_options={'user_based': True}
    )

    # 训练userCF
    userCF.fit(
        ratingDataSet.build_full_trainset()
    )

    # 预测
    # 目标用户id
    uid = 196

    # 该用户看过的所有电影ID
    watchedItemIDs = ratings[ratings['UserID'] == uid]['ItemID'].drop_duplicates().values

    # 所有电影ID
    allItemIDs = ratings['ItemID'].drop_duplicates().values

    # 保存用户和电影之间的评分
    userCF_itemIDs = []
    userCF_ratings = []

    # 遍历所有电影，拿到每部电影的ID
    for itemID in allItemIDs:
        # 如果还没看过这部电影
        if itemID not in watchedItemIDs:
            userCF_itemIDs.append(itemID)
            # 调用userCF模型的预测方法，预测用户对电影的评分
            userCF_ratings.append(userCF.predict(uid, itemID).est)

    # 结果转df
    result = pd.DataFrame({
        'userCF_itemID': userCF_itemIDs,
        'userCF_rating': userCF_ratings
    })
    # 结果按评分倒序排序
    result.sort_values(by='userCF_rating', inplace=True, ascending=False)
    print(result)
    """
              userCF_itemID  userCF_rating
    1000           1189       5.000000
    1397           1293       5.000000
    232              64       4.643135
    1200           1367       4.578842
    1454           1191       4.567010
    ...             ...            ...
    1487           1408       1.000000
    1266           1432       1.000000
    930             777       1.000000
    1134            437       1.000000
    873             314       1.000000
    """


# 使用knn实现UserCF
def itemCF_method():
    itemCF = KNNBasic(
        k=40, min_k=3,
        sim_options={'user_based': False}
    )
    itemCF.fit(ratingDataSet.build_full_trainset())
    # 目标物品ID
    itemID = 110
    # DataSet会对数据集中的物品ID重新编码，需要先找出模型中的id，inner_id
    item_inner_id = itemCF.trainset.to_inner_iid(itemID)
    # 使用inner_id进行相似物品计算，找出与该物品最接近的10个物品
    item_inner_neighbors = itemCF.get_neighbors(item_inner_id, k=10)
    # 把inner_id转换为数据集中的ID
    itemID_neighbors = [
        itemCF.trainset.to_raw_iid(inner_id)
        for inner_id in item_inner_neighbors
    ]
    print(itemID_neighbors)
    """
    [979, 919, 1211, 339, 872, 695, 903, 1115, 960, 869]
    """


if __name__ == '__main__':
    userCF_method()
    itemCF_method()
