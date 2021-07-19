#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gensim_test.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/19 13:13   SeafyLiang   1.0          gensim词向量模型学习
"""
import time
import gensim
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import LineSentence


def train_w2v_model(path):
    w2v_model = Word2Vec(LineSentence(path), workers=4, min_count=1, vector_size=300)
    w2v_model.save('w2v.model')  # 可再训练
    # w2v_model.wv.save('mmm') # 二进制，保存更快，无法再训练


def load_model():
    model = Word2Vec.load('w2v.model')
    return model


if __name__ == '__main__':
    path1 = 'small_words.txt'
    train_w2v_model(path1)
    model = load_model()
    sims = model.wv.most_similar('卫生')
    # 效果不好 原因是语料库太小
    print(sims)
    # 追加语料库，再训练模型
    new_list = []
    with open('source.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        f.close()
    for line in data:
        line = line.strip().split(' ')
        new_list.append(line)  # not extend
    model.train(corpus_iterable=new_list, epochs=1, total_examples=len(new_list))
    print(model.wv.most_similar('卫生'))
