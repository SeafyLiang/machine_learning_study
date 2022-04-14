#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   keybertDemo.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/14 18:33   SeafyLiang   1.0         keybert提取关键词及关键词短语
"""
from keybert import KeyBERT

'''
官方文档：https://maartengr.github.io/KeyBERT/
KeyBERT是一种小型且容易上手使用的关键字提取技术，它利用BERT嵌入来创建与文档最相似的关键词和关键字短语。
'''
doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs.[1] It infers a
         function from labeled training data consisting of a set of training examples.[2]
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).
      """
# model = KeyBERT('distilbert-base-nli-mean-tokens')
model = KeyBERT(model='all-mpnet-base-v2')

keywords = model.extract_keywords(doc, keyphrase_ngram_range=(1, 2))
print(keywords)
# [('supervised learning', 0.6342), ('learning algorithm', 0.6178), ('learning machine', 0.6139), ('labeled training', 0.5964), ('learning function', 0.5785)]

doc = """
         KeyBERT 是 一种 小型 且 容易 上手 使用 的 关键字 提取 技术，它 利用 BERT 嵌入 来 创建 与 文档 最 相似 的 关键词 和 关键字 短语 。
      """
# model = KeyBERT(model='distilbert-base-nli-mean-tokens')
model = KeyBERT(model='all-mpnet-base-v2')  # 根据作者的说法，all-mpnet-base-v2模型是最好的。

# 要提取关键字短语，只需将关键字短语_ngram_range设置为（1，2）或更高，具体取决于我们希望在生成的关键字短语中使用的单词数：
keywords = model.extract_keywords(doc, keyphrase_ngram_range=(1, 3))
print(keywords)
# model='distilbert-base-nli-mean-tokens'
# [('小型 容易 上手', 0.8317), ('文档 相似 关键词', 0.8259), ('创建 文档 相似', 0.8154), ('上手 使用 关键字', 0.8127), ('容易 上手 使用', 0.7997)]
# model='all-mpnet-base-v2'
# [('利用 bert 嵌入', 0.6753), ('bert 嵌入 创建', 0.6707), ('技术 利用 bert', 0.6643), ('keybert 一种 小型', 0.6404), ('keybert 一种', 0.6227)]
