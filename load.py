import tensorflow as tf 
from datetime import datetime
import numpy as np
from data_preproposs import get_data
# from model_pluscap import ImaLoc 
from model_seq import ImaLoc 

tf.reset_default_graph()
text_fold = '../../../../dailymail/story-texts'
caps_fold = '../../../../dailymail/captions'
imgs_fold = '../../../../dailymail/images'


# epoch = tf.constant(0)
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        model = ImaLoc(100, 200, 30, 100, 3, 0.001)
        train_op = model.train()
        inaccury = model.accury()
        train_start = datetime.now()
        saver = tf.train.Saver()
        saver.restore(sess, './../models/sequence/att_post4_lessthan5/50000')



        test_start = datetime.now()
        print('\n{} Test initializing'.format(test_start))
        epoch = 0
        hard = list()
        soft = list() 
        print('========训练=========')
        for line in open('../../../../dailymail/cut-trains-less5','r'):
            filename = line.strip()
            if epoch % 10 == 3 and epoch <= 50000:
                feed_value = model.get_feed_dict(text_fold, caps_fold, imgs_fold, filename, 1)
                if feed_value:              # 数据不合格时跳过  
                 
                    hard_mistakerate,soft_mistakerate,bias  = sess.run(inaccury,feed_dict = feed_value)
                    hard.append(hard_mistakerate)
                    soft.append(soft_mistakerate)
                                                    
            epoch = epoch + 1
        test_endding = datetime.now()
     
        print((np.mean(hard)))
        print(np.mean(soft))
        
#         epoch = 0
#         hard.clear()
#         soft.clear()
#         print('=======测试==========')
#         for line in open('../../../../dailymail/cut-trains-less5','r'):
#             filename = line.strip()
#             if epoch %10 == 9 and epoch >= 50000:
#                  
#                 feed_value = model.get_feed_dict(text_fold, caps_fold, imgs_fold, filename, 1)
#                 if feed_value:              # 数据不合格时跳过  
#            
#                     hard_mistakerate,soft_mistakerate,_  = sess.run(inaccury,feed_dict = feed_value)
#                     hard.append(hard_mistakerate)
#                     soft.append(soft_mistakerate)
#                                                      
#             epoch = epoch + 1
#         test_endding = datetime.now()
#           
#         print(np.mean(hard))
#         print(np.mean(soft))
              
#         print('\n{} Test ending'.format(test_endding))     
#         print('\n{} Total test time'.format(test_endding-test_start))  
#         print('\n{} Total train time'.format(train_endding-train_start))   
