import tensorflow as tf
import numpy as np
from model_withoutcap import ImaLoc
from datetime import datetime
from tqdm import tqdm
import random
from statistics import mean

text_fold = '***'   # folder for news texts
caps_fold = '***'   # folder for captions
imgs_fold = '***'   # folder for images

test_dir = '***'    # file for test data
train_dir = '***'   # file for training data
dev_dir = '***'     # file for dev data

batch_size = 100
def get_files(dir,size):
#     dir: filename; size: the count of data
    filenames = list()
    file_num = 0
    for line in open(dir,'r'):
        filename = line.strip()
        filenames.append(filename)
        file_num = file_num + 1
        if file_num >= size:
            break
    return filenames

# run the training program
def train(sess, model, train_op, inaccury, filenames, scope=None):
    # shuffle files
    random.shuffle(filenames)
    iter = int(len(filenames)/batch_size)
    for t in tqdm(range(iter),scope):
        # a batch of filenames
        begin = t*batch_size
        end = t * batch_size + batch_size
        if end > len(filenames):
            end = len(filenames)
        train_list = list()
        for i in range(begin,end):
            train_list.append(filenames[i])
        
        
        feed_value = model.get_feed_dict(text_fold, caps_fold, train_list)
        sess.run(train_op,feed_dict = feed_value)

#  calculate training error
def train_err(sess, model, inaccury, filenames, test_size_train, scope = None):
    random.shuffle(filenames)
    hard = list()
    soft = list()
    iter = int(test_size_train/batch_size)
    for t in tqdm(range(iter),scope):
        # a batch of filenames
        begin = t*batch_size
        end = t * batch_size + batch_size
        if end > test_size_train:
            end = test_size_train
        train_list = list()
        for i in range(begin,end):
            train_list.append(filenames[i])
            
        feed_value = model.get_feed_dict(text_fold, caps_fold, train_list)
        hard_mistakerate,soft_mistakerate,bias, pri  = sess.run(inaccury,feed_dict = feed_value)
        hard.append(mean(hard_mistakerate))
        soft.append(mean(soft_mistakerate))   
         
    return mean(hard), mean(soft)

# calculate testing error
def test_err(sess, model, inaccury, filenames, scope=None):
    random.shuffle(filenames)
    hard = list()
    soft = list()
    iter = int(len(filenames)/batch_size)
    for t in tqdm(range(iter),scope):
        # a batch of filenames
        begin = t*batch_size
        end = t * batch_size + batch_size
        if end > len(filenames):
            end = len(filenames)
        train_list = list()
        for i in range(begin,end):
            train_list.append(filenames[i])
            
        feed_value = model.get_feed_dict(text_fold, caps_fold, train_list)
        hard_mistakerate,soft_mistakerate,bias, pri  = sess.run(inaccury,feed_dict = feed_value)
        hard.append(mean(hard_mistakerate))
        soft.append(mean(soft_mistakerate))    
    return mean(hard), mean(soft)

with tf.Session() as sess:
    # construct the framework of model_withoutcap
    model = ImaLoc(wd_hidden_dim = 100,
                   st_hidden_dim = 200,
                   sen_maxlen = 30,
                   emb_size = 100,
                   win_size = 3,
                   learning_rate = 0.0001)
    train_op = model.train()
    inaccury = model.accury()
    
    # initialize the weights
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    # get train/dev filenames
    train_filenames = get_files(dir=train_dir, size=100000)
    dev_filenames = get_files(dir=dev_dir, size=1000)
    
    train_hard_list = list()
    train_soft_list = list()
    dev_hard_list = list()
    dev_soft_list = list()
    
    # epoch is 10
    for epoch in tqdm(range(10),'epo'):
        # train the model_withoutcap
        train(sess, model, train_op, inaccury, train_filenames, 'bat')
        
        # dev error
        dev_hard, dev_soft = test_err(sess, model, inaccury, dev_filenames, scope= 'dev_err')
        print('\n dev hard error: {}, soft error: {}'.format(dev_hard, dev_soft))
        
        # train error
        train_hard, train_soft = train_err(sess, model, inaccury, train_filenames, 1000,scope = 'train_err')
        print('\n train hard error: {}, soft error: {}'.format(train_hard, train_soft))
        
        dev_hard_list.append(dev_hard)
        dev_soft_list.append(dev_soft)
        train_hard_list.append(train_hard)
        train_soft_list.append(train_soft)
        if epoch % 2 == 1:
            # save the model_withoutcap
            saver.save(sess, '***/100000_{}'.format(epoch))
    
    del train_filenames, dev_filenames
    
    # test the model_withoutcap
    test_filenames = get_files(dir=test_dir, size=5000)
    hard, soft = test_err(sess, model, inaccury, test_filenames, scope= 'test_err')
    print('\n test hard error: {}, soft error: {}'.format(hard,soft))
    
    # record all kind of errors
    with open('***','w') as f:
        f.write(str(dev_hard_list))
        f.write('\n')
        f.write(str(dev_soft_list))
        f.write('\n')
        f.write(str(train_hard_list))
        f.write('\n')
        f.write(str(train_soft_list))
        f.write('\n')
        f.write(str([hard]))
        f.write('\n')
        f.write(str([soft]))
        


    