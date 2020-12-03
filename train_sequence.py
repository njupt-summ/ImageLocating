import tensorflow as tf
import numpy as np
# from model_seq import ImaLoc 
from model_pluscap import ImaLoc 
from datetime import datetime


text_fold = '../../../../dailymail/story-texts'
caps_fold = '../../../../dailymail/captions'
imgs_fold = '../../../../dailymail/images'
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        model = ImaLoc(100, 200, 30, 100, 3, 0.001)
        train_op = model.train()
        inaccury = model.accury()
        train_start = datetime.now()
#         saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        saver = tf.train.Saver()
        print('\n{} Train initializing'.format(train_start))
        sess.run(tf.global_variables_initializer())
        epoch = 0 
        for line in open('../../../../dailymail/cut-trains-less5','r'):
             
#             saver.save(sess, '../models/sequence/att_cap4_lessthan5_noimgseq/50000')
            filename = line.strip()
            feed_value = model.get_feed_dict(text_fold, caps_fold, imgs_fold, filename, 1)
            if feed_value:              # 数据不合格时跳过
#                 try:
                [_,loss], mistake,simi  = sess.run([train_op,inaccury, model.logits],feed_dict = feed_value)
#                 except Exception:
#                     with open('errordata','w') as f:
#                         f.write(filename)
#                         f.write('\n')       # 输出问题数据
#                     pass     
                if epoch%100 == 0:      # 隔100次查看正确情况
                    print('========')
                    print(epoch)
                    print(loss)                                            
                    print(mistake) 
                    print(simi) 
      
            epoch = epoch + 1 
            if epoch == 50000:
                break  
        train_endding = datetime.now()
        print('\n{} Train ending'.format(train_endding))     
        print('\n{} Total train time'.format(train_endding-train_start))  
#         saver.save(sess, '../models/sequence/att_cap4_lessthan5_noimgseq/50000')  
         
         
        test_start = datetime.now()
        print('\n{} Test initializing'.format(test_start))
        epoch = 0
        hard = list()
        soft = list()
        print('========训练=========')
        for line in open('../../../../dailymail/cut-trains-less5','r'):
            filename = line.strip()
            if epoch % 10 == 2:
                feed_value = model.get_feed_dict(text_fold, caps_fold, imgs_fold, filename, 1)
                if feed_value:              # 数据不合格时跳过  
          
                    hard_mistakerate,soft_mistakerate,bias  = sess.run(inaccury,feed_dict = feed_value)
                    hard.append(hard_mistakerate)
                    soft.append(soft_mistakerate)
                                             
            epoch = epoch + 1
        test_endding = datetime.now()
        
        print(np.mean(hard))
        print(np.mean(soft))
     
#         with open('error训练att_cap4_lessthan5_noimgseq','w') as f:
#             f.write(str(hard))
#             f.write(str(soft))
#             f.write(str(np.mean(hard)))
#             f.write('\n')
#             f.write(str(np.mean(soft)))
             
            
            
            
        epoch = 0
        hard.clear()
        soft.clear()
        print('=======测试2==========')
        for line in open('../../../../dailymail/cut-tests-less5','r'):
            filename = line.strip()
                
            feed_value = model.get_feed_dict(text_fold, caps_fold, imgs_fold, filename, 1)
            if feed_value:              # 数据不合格时跳过  
      
                hard_mistakerate,soft_mistakerate,_  = sess.run(inaccury,feed_dict = feed_value)
                hard.append(hard_mistakerate)
                soft.append(soft_mistakerate)
                                      
        test_endding = datetime.now()
         
        print(np.mean(hard))
        print(np.mean(soft))
#         with open('error测试att_cap4_lessthan5_noimgseq','w') as f:
#             f.write(str(hard))
#             f.write(str(soft))
#             f.write(str(np.mean(hard)))
#             f.write('\n')
#             f.write(str(np.mean(soft)))
             
        print('\n{} Test ending'.format(test_endding))     
        print('\n{} Total test time'.format(test_endding-test_start))  
        print('\n{} Total train time'.format(train_endding-train_start))   
    
        
