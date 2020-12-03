import tensorflow as tf
import numpy as np
from data_preproposs import get_data
from layers import *
import tensorflow.contrib.rnn as rnn



class ImaLoc:
    def __init__(self, wd_hidden_dim, st_hidden_dim, sen_maxlen, emb_size, win_size, learning_rate):
        self.wd_hidden_dim = wd_hidden_dim
        self.st_hidden_dim = st_hidden_dim
        self.sen_maxlen = sen_maxlen
        self.emb_size = emb_size
        self.win_size = win_size
        self.learning_rate = learning_rate
        
        self.text = tf.placeholder(shape=(None, self.sen_maxlen, self.emb_size), dtype=tf.float32, name='text')
#         self.caps = tf.placeholder(shape=(None, self.sen_maxlen, self.emb_size), dtype=tf.float32, name='caption')
        self.imgs = tf.placeholder(shape=(None, 4096), dtype=tf.float32, name='img')
        self.labels = tf.placeholder(shape=(None), dtype=tf.int64, name='locations')
        self.text_len = tf.placeholder( dtype=tf.int64, name='text_len')
        self.caps_num = tf.placeholder( dtype=tf.int64, name='tcaps_num')
        
        with tf.variable_scope('ImageLo'):
            self.word_encoder()
            self.imgs_encoder()
            self.sent_encoder()
            self.classifier()


            
    def word_encoder(self):
        with tf.variable_scope('word') as scope:
            cell_fw = rnn.GRUCell(self.wd_hidden_dim)
            cell_bw = rnn.GRUCell(self.wd_hidden_dim)

            _, state_text = bidirectional_rnn(
                cell_fw=cell_fw, 
                cell_bw=cell_bw,
                inputs=self.text, 
                scope=scope
                )
#             _, state_caps = bidirectional_rnn(
#                 cell_fw=cell_fw,
#                 cell_bw=cell_bw, 
#                 inputs=self.caps,
#                 scope=scope
#                 )
            
            self.word_text_rnn_outputs = state_text
#             self.word_caps_rnn_outputs = state_caps
    def imgs_encoder(self):
        with tf.variable_scope('imgs') as scope:
            W_img = tf.get_variable('W_img', [4096, self.wd_hidden_dim*2], tf.float32,
                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False)) 
            b_img = tf.get_variable('b_img', [self.wd_hidden_dim*2], tf.float32,
                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False)) 
            self.imgs_vec = tf.tanh(tf.matmul(self.imgs,W_img)+b_img)    
            

    def sent_encoder(self):
        with tf.variable_scope('sentence') as scope:

            cell_fw = rnn.GRUCell(self.st_hidden_dim)
            cell_bw = rnn.GRUCell(self.st_hidden_dim)

            sent_text_rnn_outputs, _ = bidirectional_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=tf.expand_dims(self.word_text_rnn_outputs,axis=0),
                scope=scope
                )
            imgs_rnn_outputs, _ = bidirectional_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=tf.expand_dims(self.imgs_vec,axis=0),
                scope=scope
                )
            
#             sent_caps_rnn_outputs, _ = bidirectional_rnn(
#                 cell_fw=cell_fw, 
#                 cell_bw=cell_bw,
#                 inputs=tf.expand_dims(self.word_caps_rnn_outputs,axis=0),
# #                 initial_state_fw=init_state_fw,
# #                 initial_state_bw=init_state_bw,
#                 scope=scope
#                 )
            self.sent_text = tf.tanh(sent_text_rnn_outputs[0])
            self.img_seq = tf.tanh(imgs_rnn_outputs[0])
#             self.sent_caps = sent_caps_rnn_outputs[0]
            
    def classifier(self):
        with tf.variable_scope('classifier'):
            W_simi = tf.get_variable('W_simi', [self.st_hidden_dim*2, 40], tf.float32,
                                      initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            U_simi = tf.get_variable('U_simi', [self.st_hidden_dim*2, 40], tf.float32, 
                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            V_simi = tf.get_variable('V_simi', [40,1], tf.float32, initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            b_simi = tf.get_variable('b_simi', [40], tf.float32, initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            
            W_mid = tf.get_variable('W_mid', [self.st_hidden_dim*4,self.st_hidden_dim*2], tf.float32, 
                                    initializer = tf.contrib.layers.xavier_initializer(uniform=False))      #方式4
#             W_mid = tf.get_variable('W_mid', [self.st_hidden_dim*2,self.st_hidden_dim*2], tf.float32, 
#                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            b_mid = tf.get_variable('b_mid', [self.st_hidden_dim*2], tf.float32,
                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            
            filter = tf.get_variable('filter', [self.win_size,1,1,1],tf.float32,
                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            
            context = self.sent_text
            
            # 循环实现sequence结构
            out = tf.Variable([])       # 暂存预测数据，以一维张量形式
            point = tf.constant(0, tf.int64)        # 循环时作为指针从imgs中取数据
            def cond(i, n, context, out):
                return  i < n
            def body(i, n,context, out):
#                 simi = post_similarity(context, self.img_seq[i], filter)        # 先卷积，再计算相似度
                simi = post_similarity(context, self.img_seq[i], filter,       # 先卷积，再计算相似度,
                                    W_simi, U_simi,V_simi,b_simi) 
#                 simi = pre_similarity(context, self.img_seq[i], self.win_size)        # 计算相似度，并紧跟着池化
#                 simi = pre_simi_att(context, self.img_seq[i], self.win_size,
#                                     W_simi, U_simi,V_simi,b_simi)        # 计算相似度，并紧跟着池化
                t = tf.argmax(simi)
                context = update_context(context, self.img_seq[i] ,simi, t, W_mid, b_mid)     # 更新插入图片后的上下文
                out = tf.concat([out, simi], axis = 0)          # 连接每一步输出值
                i = i + 1
                return i, n, context, out
            _,_,_,output = tf.while_loop(cond, body, [point, self.caps_num, context, out],
                                   shape_invariants=[point.get_shape(),self.caps_num.get_shape(),
                                                     context.get_shape(), tf.TensorShape([None])])
            self.logits = tf.reshape(output, [-1,self.text_len],name = 'similar_all')        # 将输出重新转换成矩阵形式
            
    def train(self):     
        labels = tf.one_hot(self.labels, depth=tf.cast(self.text_len,tf.int32))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits))
#         loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
        train_op = tf.train.AdamOptimizer(self.learning_rate,name='Adam').minimize(loss)    
        return  train_op, loss
                  
    def accury(self):
        pri_index = tf.argmax(self.logits, axis = 1)
        bias = tf.subtract(pri_index, self.labels, name='gap')
        incorrect_num = tf.count_nonzero(bias)
        incorect_rate_hard = tf.divide(incorrect_num,self.caps_num, name='hard_err')
        incorect_rate_soft = tf.reduce_mean(tf.abs(bias),name='soft_err')
        return incorect_rate_hard, incorect_rate_soft, bias
            
    def get_feed_dict(self, text_dir, captions_dir, img_dir, filename, flag):
#         text,caps,la = get_data(text_dir, captions_dir, img_dir, filename, flag)
        text,imgs,la = get_data(text_dir, captions_dir, img_dir, filename, flag)
#         if caps == False:
        if imgs == False:
            return False
        fd = {
            self.text: text,
            self.imgs: imgs,
#             self.caps: caps,
            self.labels: la,
            self.text_len: len(text),
#             self.caps_num: len(caps)
            self.caps_num: len(imgs)
            }
        return fd


# with tf.Session() as sess:
# # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     model = ImaLoc(100, 200, 30, 100, 3, 0.001)
#     train_op = model.train()
#     inaccury = model.accury()
#     sess.run(tf.global_variables_initializer())
#     t = 0
# #     for line in open('../../../../dailymail/trains','r'):
# #         filename = line.strip()      
# #         print(filename)      
# #         feed_value = model.get_feed_dict('../../../../dailymail/story-texts', '../../../../dailymail/captions', '../../../../dailymail/images', filename, 1)
# #       
# #         hard_mistakerate,soft_mistakerate,bias  = sess.run(inaccury,feed_dict = feed_value)
# #          
#     feed_value= model.get_feed_dict('../../../../dailymail/story-texts', '../../../../dailymail/captions', '../../../../dailymail/images',
#                                      'e6bb8008e0c47cd5ca1810e01e364986348a413e',1)
#     for l in range(40000):
#         t = t+1
#         print(t)
#         [_,loss],acc = sess.run([train_op,inaccury],feed_dict = feed_value)
# # # #         if l%100 == 0:
# # #             print('========')
# # # #             print(l)
#         print(loss)
#         print(acc)
# # #             
#     print(sess.run([inaccury, model.pri_index],feed_dict = feed_value))


