
# coding: utf-8

# *** Unuspervised Text Feature Extraction ***

# In[1]:

import os
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import jieba
import jieba.analyse
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import ast

import random

import logging
import html_template as ht
import collections
from collections import OrderedDict
import itertools
import math

from itertools import combinations
from itertools import permutations

    
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


pd.set_option('display.max_columns', None)
punct = u'''\n +-％%:!),.:;?]}¢'"、。〉》」』】〕〗〞︰|︱︳丨﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘—_…~/#><'''
jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('userdict.txt')


# 

# In[2]:

def remove_duplicate(powerterm):
    remove_list=[]
    for tup in powerterm:
        tmp2 = powerterm.copy()
        tmp2.remove(tup)
        tmp2 = dict(tmp2)
        result = [(key, value) for key, value in tmp2.items() if tup[0] in key]
        if len(result)==0:
            continue
        result = sorted(result, key=lambda tup: tup[1], reverse=True)
        if tup[1] < result[0][1]:
            remove_list.append(tup)

    _ = list(map(lambda tup:powerterm.remove(tup), remove_list))    


# In[3]:

#計算分數的function
def NC_Scoring(tmp, nature_chunk):
    score =0
    subscore_fb=subscore_fl=subscore_fr=subscore_fn=0
    for chunk in nature_chunk:
        if tmp == chunk:
            subscore_fb = subscore_fb+1
        if chunk.startswith(tmp):
            subscore_fl = subscore_fl+1
        if chunk.endswith(tmp):
            subscore_fr = subscore_fr+1
        if tmp in chunk:
            subscore_fn = subscore_fn+1
    score = 0.5*subscore_fb + 0.2*(subscore_fl+subscore_fr)+0.1*subscore_fn
    score = len(tmp) * score
    return float('%.3f'% score)

#NC_Scoring('TITAN X')


# In[4]:

def getPowerTerm(powerterm, paragraph, nature_chunk):
    #1. 利用自然標注信息＆已知詞, 斷字段詞
    corpora_sentence = []
    for p in paragraph:
        term_known = list(filter(lambda t: t[0] in p, powerterm))
        term_known = list(map(lambda tup: tup[0], term_known))
        for w in powerterm:
            p = p.replace(w[0], '')
        p_cut = list(p)
        p_cut = list(filter(lambda x: x not in punct, p_cut))
        corpora_sentence.append(p_cut+term_known)
    
    #2. LDA
    sen_dictionary = gensim.corpora.Dictionary(corpora_sentence)  
    sen_corpus = [sen_dictionary.doc2bow(text) for text in corpora_sentence]  

    K = 5
    lda = gensim.models.ldamodel.LdaModel(corpus=sen_corpus, id2word=sen_dictionary, num_topics=K, update_every=0, passes=1)  
    corpus_lda = lda[sen_corpus] 
    wordTopic = lda.print_topics(num_topics=-1, num_words=15)
    wordTopic_pure = (list(map(lambda tup: (list(map(lambda tup: tup.split('*')[1].replace('"','').strip(),tup[1].split('+')))   ),wordTopic)))

    #3. 產生NxN term candidate, scoring排序, 取前5%為power term
    wordTopic_NN = list(map(lambda x: list(map(lambda tup: tup[0]+tup[1], list(permutations(x, 2)))) , wordTopic_pure))
    powerterm_candidate = set(sum(wordTopic_NN,[]))
    powerterm = list(map(lambda term: (term, NC_Scoring(term, nature_chunk)), powerterm_candidate))
    powerterm = list(filter(lambda tup: tup[1]>0,powerterm))
    powerterm = sorted(powerterm, key=lambda tup: tup[1], reverse=True)
    #powerterm = powerterm[: int(len(powerterm)*0.1)]
    #powerterm = list(map(lambda tup: tup[0], powerterm))
    return powerterm


# In[7]:

def power_term(doc, itera = 20):
    #產生nature_chunk
    paragraph = list(filter(lambda x: x != '...', doc.split('\n')))
    nature_chunk = []
    for p in paragraph:
        chunk = ''.join((char if char.isalpha() or char.isdigit() or char.isspace() else '|') for char in p).strip().split('|')
        chunk = list(map(lambda x: x.strip(), chunk))
        nature_chunk = nature_chunk+chunk
    nature_chunk = list(filter(lambda a: a != '', nature_chunk))
    
    powerterm = []
    powerterm_word=[]
    for i in range(itera):
        powerterm_new = getPowerTerm(powerterm, paragraph,nature_chunk)
        powerterm_new = list(filter(lambda x: x[0] not in powerterm_word, powerterm_new))
        powerterm = powerterm + powerterm_new
        powerterm = sorted(powerterm, key=lambda tup: tup[1], reverse=True)
        remove_duplicate(powerterm)
        powerterm_word = list(map(lambda tup: tup[0], powerterm))
    return powerterm



# In[8]:

doc = '【IT168 資訊】為了解決當今世界最尖端的技術挑戰之一，NVIDIA剛剛推出了全新的硬體和軟體，將前所未有地提高深度學習研究的速度、易用性和功用。\n在人工智慧領域快速成長的深度學習技術是一項創新的計算引擎，可應用在從先進醫藥研究到全自動駕駛汽車的多元領域。\n...\nNVIDIA聯合創始人、總裁兼執行長黃仁勛先生在GPU技術大會的開幕主題演講活動上，對在座的四千名與會嘉賓展示三項將推動深度學習的新技術：\n·NVIDIA GeForce GTX TITAN X - 為訓練深度神經網絡而開發的最強大的處理器。\n·DIGITS深度學習GPU訓練系統 - 數據科學家與研究人員能利用這套軟體便捷地開發出高品質深度神經網絡。\n·DIGITS DevBox - 全球最快的桌邊型深度學習工具 - 專為相關任務而打造，採用TITAN X GPU，搭配直觀易用的DIGITS訓練系統。\nGeForce GTX TITAN X的另一面\nTITANX是NVIDIA全新推出的旗艦級遊戲顯卡，但也特別適合用於深度學習。\n...\n在舊金山舉辦的遊戲開發者大會上讓各位先睹為快了TITAN X的身影，它以電影《霍比特人》里的史矛戈巨龍為藍本，播放了一段名為《暗影神偷》精彩的虛擬現實體驗。\n...\n在TITANX上能以4K的超高畫質呈現最新AAA遊戲大作的瑰麗畫面，可以在開啟FXAA高設定值的情況下，以每秒40幀(40FPS)運行《中土世界：暗影魔多》(Middle-earth:Shadow of Mordor)遊戲，而在九月發行的GeForce GTX 980上則是以30FPS來運行。\n採用NVIDIA Maxwell GPU架構的TITAN X，結合3072個處理核心、單精度峰值性能為7 teraflops，加上板載的12GB顯存，在性能和性能功耗比方面皆是前代產品的兩倍。\n憑藉強大的處理能力和336.5GB/s的帶寬，讓它能處理用於訓練深度神經網絡的數百萬的數據。例如，TITAN X在工業標準模型AlexNet上，花了不到三天的時間、使用120萬個ImageNet圖像數據集去訓練模型，而使角16核心的CPU得花上四十多天。\n現已上市的GeForce GTX TITAN X售價為7999元人民幣。\nDIGITS：通往最佳深度神經網絡的便捷之路\n使用深度神經網絡來訓練電腦教自己如何分類和識別物體，是一件繁重又費時的事情。\nDIGITS深度學習 GPU 訓練系統軟體自始至終都將為用戶提供所需數據，幫助用戶建立最優的深度神經網絡，改變上述的局面。\n訪問http://developer.nvidia.com/diqits即可下載DIGITS深度學習GPU訓練系統，這是首套用於設計、訓練和驗證圖像分類深度神經網絡的多合一圖形系統。\nDIGITS可在安裝、配置和訓練深度神經網絡過程中為用戶提供指導一處理複雜的工作好讓科學家能專心在研究活動和結果上。\n得益於其直觀的用戶介面和強大的工作流程管理能力，不論是在本地系統還是在網絡上使用DIGITS，準備和加載訓練數據集都相當簡單。\n這是同類系統中首個提供實時監控和可視化功能的系統，用戶可以對工作進行微調。它還支持GPU加速版本Caffe，目前，這一框架在眾多數據科學家和研究人員中都得到了廣泛使用，用於構建神經網絡。\nDIGITS DevBox：全球最快的桌邊型深度學習機器\nNVIDIA深度學習工程團隊為了自己的研發工作而開發的DIGITS DevBox，是一套集多項功能於一身的平台，能夠加快深度學習的研究活動。\n...\n它採用四個TITAN X GPU、從內存到I/O，DevBox的每個組件都進行了最佳化調試，可為最嚴苛的深度學習研究工作提供高效率的性能表現。\n它己經預先安裝了數據科學家和研究人員在開發自己的深度神經網絡時，所需要使用到的各種軟體，包括DIGITS軟體包、最受歡迎的深度學習架構一Caffe、Theano和Torch，還有NVIDIA完整的GPU加速深度學習庫cuDNN 2.0。\n所有這些都集結在這個高能效、靜默、運行流暢且外形優美的軟體包中，只需要普通的電源插座，低調安置在您的桌下即可。\n較早期的多GPU訓練成果顯示，在關鍵深度學習測試中，DIGITS DevBox可以提供4倍於單個TITAN X的性能。使用DIGITS DevBox來訓練AlexNet只要13個小時就能完成，而使用最好的單GPU PC的話則是兩天，單純使用CPU系統的話則要一個月以上的時間。'   
powerterm = power_term(doc)


# In[9]:

powerterm

