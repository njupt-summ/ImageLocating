import nltk
import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import norm
from numpy import argmax
from VGG19 import get_imgs_fea
import os

vec_model = KeyedVectors.load("../models/word2vec/word_dm.model_withoutcap")   # loading pre-trained word2vec model_withoutcap
stop_words = nltk.corpus.stopwords.words('english')
vec_len = 100       # the dimension of word embedding
sen_maxlen = 30     # the number of words in sentence
pad = [1e-10 for _ in range(vec_len)]       # padding vector for sentence
pad_txt = [[0 for _ in range(vec_len)] for _ in range(sen_maxlen)]      
pad_img = [1e-10 for _ in range(4096)]      # the dimension of image vector is 4096

# read text and splitting it
def text_read(file_path):
    with open(file_path ,'r') as f:
        text = f.read().split('\n')
    parag = list()
    for sent in text:
        word = nltk.word_tokenize(sent)
        filter_word = [w for w in word if not w in stop_words]  # a sentence without stopwords
        if filter_word != []:
            parag.append(filter_word)
    return parag
 
# every sentence vectors are summation of word embedding
def text_vec_add(text):
    text_list = list()
    for sent in text:
        v = np.zeros(vec_len)
        for word in sent:
            try:
                # avoid some words not in word2vec's word list
                vec = vec_model[word]   # load word embedding
                v += vec
            except Exception:
                pass     
        text_list.append(v)
    return text_list
    # [sentence_number, word_embedding_size]
 
# pad for sentence
def text_vec_pad(text):
    text_list = list()
    sent_len = list()
    for sent in text:
        sent_list = list()
        for word in sent:
            try:                
                vec = vec_model[word]
                sent_list.append(list(vec))
            except Exception:
                pass
        length = len(sent_list)
#         make sure the length of every sentence vector is fixed
        if length > sen_maxlen:
            text_list.append(sent_list[:sen_maxlen])
        elif length < sen_maxlen :
            gap = sen_maxlen - sent_len
            text_list.append(sent_list + [pad for _ in range(gap)])
        else:
            text_list.append(sent_list)
        sent_len.append(length) 
             
    return text_list, sent_len
    # [sentence_number, sen_maxlen, word_embedding_size]
 
 
# cosine distance
def vector_similarity(s1, s2):
    if (s1 == 0).all():
        s1 = pad
    if (s2 == 0).all():
        s2 = pad
    return np.dot(s1, s2) / (norm(s1) * norm(s2))
 
# find the location of the picture according to the caption
def caps_correspond_locat(caps,text):
    caps_vec = text_vec_add(caps)
    text_vec = text_vec_add(text)
    locats = list()
        
    for i in range(len(caps_vec)):
        l = list()
        for s in text_vec:
#             the cosine distance between image i and each sentence
            l.append(vector_similarity(s, caps_vec[i]))
#         choose the biggest value as the ground truth
        locats.append(argmax(l))
    return locats
     
# the back position value shounle be reduced, after the front caption are deleted
def deal_text_label(caps_locat_list):  
    rm_repet = sorted(list(set(caps_locat_list)))
    gap = [rm_repet.index(caps_locat_list[i]) for i in range(len(caps_locat_list))]
    post_cut_locat_list = np.array(caps_locat_list) - np.array(gap)
    return  post_cut_locat_list

# load data for every news
def get_data(text_dir, caps_dir, imgs_dir, file_name):
#     read news text, caption and image vector processed by VGGNet
    caps_path = os.path.join(caps_dir,file_name)
    text_path = os.path.join(text_dir,file_name)
    imgs_path = os.path.join(imgs_dir,file_name)
    text = text_read(text_path)
    caps = text_read(caps_path)
    imgs = get_imgs_fea(imgs_path)
#     get image's location according to caption
    caps_locat_list = caps_correspond_locat(caps,text)
     
    # delete captions from news text
    rm_repet = sorted(list(set(caps_locat_list)))
    rm_repet.reverse()
    for i in rm_repet:
        del text[i]
    caps_locat_list = deal_text_label(caps_locat_list)
    txt_pad, txt_len = text_vec_pad(text)
    cap_pad, cap_len = text_vec_pad(caps)
    return txt_pad, txt_len, cap_pad, cap_len, imgs, caps_locat_list


# load data for batch news and normalize them
def batch_normalize(text_dir, captions_dir, img_dir, filenames):
    text_batch = list()
    labels_batch = list()
    sent_len_batch = list()     # the set of lengths of every sentence in news text
    imgs_batch = list()
    cap_len_batch = list()      # the set of lengths of every caption in captions text
    caps_batch = list()
    
    for f in filenames:
        text, caps, sent_len, cap_len, img, positions = get_data(text_dir, captions_dir, img_dir, f)
        labels_batch.append(positions)
        imgs_batch.append(img)
        text_batch.append(text)
        sent_len_batch.append(sent_len)
        cap_len_batch.append(cap_len)
        caps_batch.append(caps)
    
#     normalize the training data
    text_batch_norm = list()
    caps_batch_norm = list()
    imgs_batch_norm = list()
    labels_batch_norm = list()
    sent_len_batch_norm = list()
    cap_len_batch_norm = list()
    text_len = [len(text) for text in text_batch]   # the set of lengths of news text
    caps_num = [len(cap) for cap in caps_batch]     # the set of lengths of captions text
    
#     pad data of every item to the maximum
    text_size = max(text_len)
    caps_size = max(caps_num)
    
    batch_size = len(text_batch)
    for i in range(batch_size):
        if len(text_batch[i]) < text_size:
            gap = text_size - len(text_batch[i])
            text_batch_norm.append(text_batch[i] + [pad_txt for _ in range(gap)])
            sent_len_batch_norm.append(sent_len_batch[i] + [0 for _ in range(gap)])
        else:
            text_batch_norm.append(text_batch[i])
            sent_len_batch_norm.append(sent_len_batch[i])
             
        if len(caps_batch[i]) < caps_size:
            gap = caps_size - len(caps_batch[i])
            caps_batch_norm.append(caps_batch[i] + [pad_txt for _ in range(gap)])
            labels_batch_norm.append(labels_batch[i] + [0 for _ in range(gap)])
            cap_len_batch_norm.append(cap_len_batch[i] + [0 for _ in range(gap)])
            imgs_batch_norm.append(imgs_batch[i] + [pad_img for _ in range(gap)])
        else:
            caps_batch_norm.append(caps_batch[i])
            labels_batch_norm.append(labels_batch[i])
            cap_len_batch_norm.append(cap_len_batch[i])
            imgs_batch_norm.append(imgs_batch[i])
         
    del text_batch, caps_batch, imgs_batch, labels_batch, sent_len_batch, cap_len_batch
    return text_batch_norm, caps_batch_norm, imgs_batch_norm, labels_batch_norm, sent_len_batch_norm, cap_len_batch_norm, text_len, caps_num

