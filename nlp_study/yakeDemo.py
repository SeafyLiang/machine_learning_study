#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   yakeDemo.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/15 21:27   SeafyLiang   1.0       yake提取关键词和短语
"""
'''
它是一种轻量级、无监督的自动关键词提取方法，它依赖于从单个文档中提取的统计文本特征来识别文本中最相关的关键词。该方法不需要针对特定的文档集进行训练，也不依赖于字典、文本大小、领域或语言。Yake 定义了一组五个特征来捕捉关键词特征，这些特征被启发式地组合起来，为每个关键词分配一个分数。分数越低，关键字越重要。你可以阅读原始论文[2]，以及yake 的Python 包[3]关于它的信息。
论文: https://www.sciencedirect.com/science/article/abs/pii/S0020025519308588
yake包: https://github.com/LIAAD/yake
'''
import yake

title = "VECTORIZATION OF TEXT USING DATA MINING METHODS"
text = "In the text mining tasks, textual representation should be not only efficient but also interpretable, as this enables an understanding of the operational logic underlying the data mining models. Traditional text vectorization methods such as TF-IDF and bag-of-words are effective and characterized by intuitive interpretability, but suffer from the «curse of dimensionality», and they are unable to capture the meanings of words. On the other hand, modern distributed methods effectively capture the hidden semantics, but they are computationally intensive, time-consuming, and uninterpretable. This article proposes a new text vectorization method called Bag of weighted Concepts BoWC that presents a document according to the concepts’ information it contains. The proposed method creates concepts by clustering word vectors (i.e. word embedding) then uses the frequencies of these concept clusters to represent document vectors. To enrich the resulted document representation, a new modified weighting function is proposed for weighting concepts based on statistics extracted from word embedding information. The generated vectors are characterized by interpretability, low dimensionality, high accuracy, and low computational costs when used in data mining tasks. The proposed method has been tested on five different benchmark datasets in two data mining tasks; document clustering and classification, and compared with several baselines, including Bag-of-words, TF-IDF, Averaged GloVe, Bag-of-Concepts, and VLAC. The results indicate that BoWC outperforms most baselines and gives 7% better accuracy on average"
full_text = title + ", " + text
print("The whole text to be usedn", full_text)

'''
首先从 Yake 实例中调用 KeywordExtractor 构造函数，它接受多个参数，其中重要的是：要检索的单词数top，此处设置为 10。参数 lan：此处使用默认值en。可以传递停用词列表给参数 stopwords。然后将文本传递给 extract_keywords 函数，该函数将返回一个元组列表 (keyword: score)。关键字的长度范围为 1 到 3。
'''
kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
keywords = kw_extractor.extract_keywords(full_text)
for kw, v in keywords:
    print("Keyphrase: ", kw, ": score", v)
'''
从结果看有三个关键词与作者提供的词相同，分别是text mining, data mining 和 text vectorization methods。注意到Yake会区分大写字母，并对以大写字母开头的单词赋予更大的权重。
'''