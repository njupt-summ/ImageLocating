import nltk
import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import norm
from numpy import argmax
from VGG19 import get_imgs_fea
import os

vec_model = KeyedVectors.load("../models/word2vec/word_dm.model")
stop_words = nltk.corpus.stopwords.words('english')
vec_len = 100
sen_maxlen = 30
pad = [1e-10 for _ in range(100)]
def text_read(file_path):
    with open(file_path ,'r') as f:
        text = f.read().split('\n')
    parag = list()
    for sent in text:
        word = nltk.word_tokenize(sent)
        filter_word = [w for w in word if not w in stop_words]
        if filter_word != []:
            parag.append(filter_word)
    return parag
 
#用于寻找下标
def text_vec_add(text):
    text_list = list()
    for sent in text:
        v = np.zeros(vec_len)
        for word in sent:
            try:
                vec = vec_model[word]
                v += vec
            except Exception:
                pass     
        text_list.append(v)
    return text_list 
 
#用于将数据feed进graph 
def text_vec_pad(text):
    text_list = list()
    for sent in text:
        sent_list = list()
        for word in sent:
            try:                # 防止单词不在word2ec表里面
                vec = vec_model[word]
                sent_list.append(list(vec))
            except Exception:
                pass
        sent_len = len(sent_list)
        if sent_len > sen_maxlen:       # 将句子长度固定在sent_len
            text_list.append(sent_list[:sen_maxlen])
        elif sent_len < sen_maxlen :
            gap = sen_maxlen - sent_len
            text_list.append(sent_list + [pad for _ in range(gap)])
        else:
            text_list.append(sent_list)
             
    return text_list
 
 
#杰卡德系数
def vector_similarity(s1, s2):
    
    if (s1 == 0).all():
        s1 = pad
    if (s2 == 0).all():
        s2 = pad
    return np.dot(s1, s2) / (norm(s1) * norm(s2))
 
#caption在原文的位置下标
def caps_correspond_locat(caps,text):
    caps_vec = text_vec_add(caps)
    text_vec = text_vec_add(text)
    locats = list()
#     print(caps_vec[8])
#     for s in text_vec:
#         print(np.array(vector_similarity(s, caps_vec[9])).shape)
        
        
    for i in range(len(caps_vec)):
        l = list()
        for s in text_vec:
            l.append(vector_similarity(s, caps_vec[i]))
        locats.append(argmax(l))
    return locats
     
#消除删除caption对后面cap位置的影响
def deal_text_label(caps_locat_list):  
    rm_repet = sorted(list(set(caps_locat_list)))
    gap = [rm_repet.index(caps_locat_list[i]) for i in range(len(caps_locat_list))]
    post_cut_locat_list = np.array(caps_locat_list) - np.array(gap)
    return  post_cut_locat_list
 
 
#读取数据并处理
def get_data(text_dir, caps_dir, imgs_dir, file_name, insert):
    caps_path = os.path.join(caps_dir,file_name)
    text_path = os.path.join(text_dir,file_name)
    imgs_path = os.path.join(imgs_dir,file_name)
    text = text_read(text_path)
    caps = text_read(caps_path)
    imgs = get_imgs_fea(imgs_path)
    if len(caps)==0:
        return False, False, False
    if len(caps) != len(imgs):
        return False, False, False
    caps_locat_list = caps_correspond_locat(caps,text)
     
    # 将caption从原文删除
    rm_repet = sorted(list(set(caps_locat_list)))
    rm_repet.reverse()
    for i in rm_repet:
        del text[i]
    if insert:
        caps_locat_list = deal_text_label(caps_locat_list)
    return text_vec_pad(text), imgs, caps_locat_list

# 返回带有caption的数据
def get_data_cap(text_dir, caps_dir, imgs_dir, file_name, insert):
    caps_path = os.path.join(caps_dir,file_name)
    text_path = os.path.join(text_dir,file_name)
    imgs_path = os.path.join(imgs_dir,file_name)
    text = text_read(text_path)
    caps = text_read(caps_path)
    imgs = get_imgs_fea(imgs_path)
    if len(caps)==0:
        return False, False, False, False
    if len(caps) != len(imgs):
        return False, False, False, False
    caps_locat_list = caps_correspond_locat(caps,text)
     
    # 将caption从原文删除
    rm_repet = sorted(list(set(caps_locat_list)))
    rm_repet.reverse()
    for i in rm_repet:
        del text[i]
    if insert:
        caps_locat_list = deal_text_label(caps_locat_list)
    return text_vec_pad(text), text_vec_pad(caps), imgs, caps_locat_list
# count = 0
# for line in open('../../../../dailymail/trains','r'):
#     filename = line.strip()
#     file = os.path.join('../../../../dailymail/story-texts',filename)
#     text_read(file)
# 
# t,c,l = get_data('../../../../dailymail/story-texts', '../../../../dailymail/captions',
#                  'f889890ebe4087b8cc70d1095984f78092a0f964',0)
# print(l)

# 6c9732511e8fe79b7168b700c9e1a6c6ef0bcb74
# 48948a2947e8a6eae49b4931db393c747198991c
# 453c2d48410a048227852a5636b5900279768279
# 6fa40e43b522cb562576a26ec025d31c6b194480
# 4e8eb04df0b230449ee4080fb0780cfc79719ca8
# f889890ebe4087b8cc70d1095984f78092a0f964
# b412608a7f30af28fb8615e4b522b7dcecabe212




