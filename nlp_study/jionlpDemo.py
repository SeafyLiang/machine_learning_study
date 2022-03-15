#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   jionlpDemo.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/3/15 14:30   SeafyLiang   1.0       jionlp-中文 NLP 预处理工具包
"""
# pip3 install pkuseg
# pip3 install jionlp
import jionlp as jio

"""
JioNLP：中文 NLP 预处理工具包 A Python Lib for Chinese NLP Preprocessing
github：https://github.com/dongrixinyu/JioNLP
"""

# 1、关键短语抽取
text = '全球领先的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。'
key_phrases = jio.keyphrase.extract_keyphrase(text)
print(key_phrases)
# ['中文搜索', '中文网页数据库', '搜索结果', '百度', '网民']


# 2、文本摘要抽取
res = jio.summary.extract_summary(text)
print(res)
# 全球领先的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。

# 3、删除异常字符
text = '''中国人民坚强伟大√~~'''
res = jio.remove_exception_char(text)
print(res)

# '中国人民坚强伟大~~'

# 4、清洗文本
text = '''<p><br></p>       <p><span>在17日举行的十三届全国人大一次会议记者会上，环境保护部部长李干杰就“打好污染防治攻坚战”相关问题回答记者提问。李干杰表示，打好污染防治攻坚战，未来将聚焦“围绕三类目标，突出三大领域，强化三个基础”开展具体工作。</span></p><p><span>顶层设计聚焦“三个三”</span></p><p><span>党的十八大以来>，我国生态环境保护工作乃至整个生态文明建设工作，决心之大、力度之大、成效之大前所未有，取得了历史性成就，发生了历史性变革。（责任编辑：唐小林）联系电话：13302130583，邮箱：dongrixinyu.89@163.com~~~~'''
res = jio.clean_text(text)
print(res)

# ' 在17日举行的十三届全国人大一次会议记者会上，环境保护部部长李干杰就“打好污染防治攻坚战”相关问题回答记者提问。李干杰表示，打好污染防治攻坚战，未来将聚焦“围绕三类目标，突出三大领域，强化三个基础”开展具体工作。顶层设计聚焦“三个三”党的十八大以来，我国生态环境保护工作乃至整个生态文明建设工作，决心之大、力度之大、>成效之大前所未有，取得了历史性成就，发生了历史性变革。联系电话：，邮箱：~'

# 5、繁体转简体
text = '今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？'
res1 = jio.tra2sim(text, mode='char')
res2 = jio.tra2sim(text, mode='word')
print(res1)
print(res2)

# 今天天气好晴朗，想吃速食面。你还在工作吗？在太空梭上工作吗？
# 今天天气好晴朗，想吃方便面。你还在工作吗？在航天飞机上工作吗？

# 6、简体转繁体
text = '今天天气好晴朗，想吃方便面。你还在工作吗？在航天飞机上工作吗？'
res1 = jio.sim2tra(text, mode='char')
res2 = jio.sim2tra(text, mode='word')
print(res1)
print(res2)

# 今天天氣好晴朗，想喫方便面。妳還在工作嗎？在航天飛機上工作嗎？
# 今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？


# 7、汉字转拼音
text = '中华人民共和国。'
res1 = jio.pinyin(text)
res2 = jio.pinyin(text, formater='simple')
res3 = jio.pinyin('中国', formater='detail')
print(res1)
print(res2)
print(res3)

# ['zhōng', 'huá', 'rén', 'mín', 'gòng', 'hé', 'guó', '<py_unk>']
# ['zhong1', 'hua2', 'ren2', 'min2', 'gong4', 'he2', 'guo2', '<py_unk>']
# [{'consonant': 'zh', 'vowel': 'ong', 'tone': '1'},
#  {'consonant': 'g', 'vowel': 'uo', 'tone': '2'}]
