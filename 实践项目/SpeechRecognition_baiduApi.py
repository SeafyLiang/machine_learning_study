#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   SpeechRecognition_baiduApi.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/22 14:21   SeafyLiang   1.0          None
"""
from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '24588270'
API_KEY = 'TDGqU4fy70ptbezeRhMsBLkU'
SECRET_KEY = ''

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

filePath = "/Users/seafyliang/DEV/Code_projects/Python_projects/算法项目学习/PPASR--_github/dataset/output.wav"


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 识别本地文件
result = client.asr(get_file_content(filePath), 'wav', 16000, {
    'dev_pid': 1537,
})
print(str(result))