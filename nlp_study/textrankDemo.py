#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   textrankDemo.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/16 21:51   SeafyLiang   1.0       Textrank生成关键词
"""
from summa import keywords

'''
TextRank 是一种用于提取关键字和句子的无监督方法。它一个基于图的排序算法。其中每个节点都是一个单词，边表示单词之间的关系，这些关系是通过定义单词在预定大小的移动窗口内的共现而形成的。
该算法的灵感来自于 Google 用来对网站进行排名的 PageRank。它首先使用词性 (PoS) 对文本进行标记和注释。它只考虑单个单词。没有使用 n-gram，多词是后期重构的。
TextRank算法是利用局部词汇之间关系（共现窗口）对后续关键词进行排序，直接从文本本身抽取。
'''

title = "VECTORIZATION OF TEXT USING DATA MINING METHODS"
text = "In the text mining tasks, textual representation should be not only efficient but also interpretable, as this enables an understanding of the operational logic underlying the data mining models. Traditional text vectorization methods such as TF-IDF and bag-of-words are effective and characterized by intuitive interpretability, but suffer from the «curse of dimensionality», and they are unable to capture the meanings of words. On the other hand, modern distributed methods effectively capture the hidden semantics, but they are computationally intensive, time-consuming, and uninterpretable. This article proposes a new text vectorization method called Bag of weighted Concepts BoWC that presents a document according to the concepts’ information it contains. The proposed method creates concepts by clustering word vectors (i.e. word embedding) then uses the frequencies of these concept clusters to represent document vectors. To enrich the resulted document representation, a new modified weighting function is proposed for weighting concepts based on statistics extracted from word embedding information. The generated vectors are characterized by interpretability, low dimensionality, high accuracy, and low computational costs when used in data mining tasks. The proposed method has been tested on five different benchmark datasets in two data mining tasks; document clustering and classification, and compared with several baselines, including Bag-of-words, TF-IDF, Averaged GloVe, Bag-of-Concepts, and VLAC. The results indicate that BoWC outperforms most baselines and gives 7% better accuracy on average"
full_text = title + ", " + text

TR_keywords = keywords.keywords(full_text, scores=True)
print(TR_keywords[0:10])
# [('methods', 0.29585314188985434), ('method', 0.29585314188985434), ('document', 0.29300649554724484), ('concepts', 0.2597209892723852), ('concept', 0.2597209892723852), ('mining', 0.20425273810869513), ('vectorization', 0.20080655873686565), ('word vectors', 0.18267366210822228), ('computationally', 0.16718186386765732), ('computational', 0.16718186386765732)]
