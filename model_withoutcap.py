import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
from datareader import batch_normalize
from layers import *
import random

class ImaLoc:
    def __init__(self, wd_hidden_dim, st_hidden_dim, sen_maxlen, emb_size, win_size, learning_rate):
        self.wd_hidden_dim = wd_hidden_dim      # the number of cells in word-level BiRNN
        self.st_hidden_dim = st_hidden_dim      # the number of cells in sentence-level BiRNN
        self.sen_maxlen = sen_maxlen            # the number of words in sentence
        self.emb_size = emb_size                # the dimension of word embedding
        self.win_size = win_size                # the size of pooling or conv window
        self.learning_rate = learning_rate
        
        self.text = tf.placeholder(shape=(None, None, self.sen_maxlen, self.emb_size), dtype=tf.float32, name='text')
        self.imgs = tf.placeholder(shape=(None, None, 4096), dtype=tf.float32, name='img')
        self.labels = tf.placeholder(shape=(None, None), dtype=tf.int64, name='locations')
        self.sequence_len_sent = tf.placeholder(shape=(None, None), dtype=tf.int64, name='sequence_len_sent')
         
        self.sequence_len_text = tf.placeholder(shape=(None), dtype=tf.int64, name='sequence_len_text')
        self.sequence_len_capdoc = tf.placeholder(shape=(None), dtype=tf.int64, name='sequence_len_capdoc')
        
        self.text_size = tf.reduce_max(self.sequence_len_text)
        self.caps_size = tf.reduce_max(self.sequence_len_capdoc)
        
        with tf.variable_scope('ImageLo'):
            self.word_encoder()
            self.imgs_encoder()
            self.sent_encoder()
            self.classifier()


            
    def word_encoder(self):
        with tf.variable_scope('word') as scope:
            cell_fw = rnn.GRUCell(self.wd_hidden_dim)
            cell_bw = rnn.GRUCell(self.wd_hidden_dim)
            
            text_rnn_input = tf.reshape(self.text, [-1,self.sen_maxlen, self.emb_size])
            text_seq_len = tf.reshape(self.sequence_len_sent, [-1])
            
            # encode news text
            init_state_fw_t = tf.tile(tf.get_variable('word_init_state_fw', 
                                                    shape=[1, self.wd_hidden_dim], 
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
                                      multiples=[tf.shape(text_rnn_input)[0], 1])
            init_state_bw_t = tf.tile(tf.get_variable('word_init_state_bw',
                                                    shape=[1, self.wd_hidden_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
                                      multiples=[tf.shape(text_rnn_input)[0], 1])
 
            _, state_text = bidirectional_rnn(
                cell_fw=cell_fw, 
                cell_bw=cell_bw,
                inputs=text_rnn_input, 
                input_lengths= text_seq_len,
                initial_state_fw=init_state_fw_t,
                initial_state_bw=init_state_bw_t,
                scope=scope
                )
        
            self.word_text_rnn_outputs = tf.reshape(state_text,[-1, self.text_size, 2*self.wd_hidden_dim])
          
    def imgs_encoder(self):
        with tf.variable_scope('imgs') as scope:
            W_img = tf.get_variable('W_img', [4096, self.wd_hidden_dim*2], tf.float32, 
                                    initializer = tf.contrib.layers.xavier_initializer(uniform=False)) 
            b_img = tf.get_variable('b_img', [self.wd_hidden_dim*2], tf.float32, 
                                    initializer = tf.contrib.layers.xavier_initializer(uniform=False)) 
            self.imgs_vec = tf.tanh(tf.matmul(self.imgs, W_img)+b_img)


    def sent_encoder(self):
        with tf.variable_scope('sentence') as scope:
            
            cell_fw_t = rnn.GRUCell(self.st_hidden_dim)
            cell_bw_t = rnn.GRUCell(self.st_hidden_dim)
            
            init_state_fw_t = tf.tile(tf.get_variable('sent_init_state_fw_t', 
                                                    shape=[1, self.st_hidden_dim], 
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
                                      multiples=[tf.shape(self.word_text_rnn_outputs)[0], 1])
            init_state_bw_t = tf.tile(tf.get_variable('sent_init_state_bw_t',
                                                    shape=[1, self.st_hidden_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
                                      multiples=[tf.shape(self.word_text_rnn_outputs)[0], 1])


            sent_text_rnn_outputs, _ = bidirectional_rnn(
                cell_fw=cell_fw_t,
                cell_bw=cell_bw_t,
                inputs=self.word_text_rnn_outputs,
                input_lengths= self.sequence_len_text,
                initial_state_fw=init_state_fw_t,
                initial_state_bw=init_state_bw_t,
                scope=scope
                )
        with tf.variable_scope('image') as scope:

            cell_fw_c = rnn.GRUCell(self.st_hidden_dim)
            cell_bw_c = rnn.GRUCell(self.st_hidden_dim)           
            init_state_fw_c = tf.tile(tf.get_variable('sent_init_state_fw_c', 
                                                    shape=[1, self.st_hidden_dim], 
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
                                      multiples=[tf.shape(self.imgs_vec)[0], 1])
            init_state_bw_c = tf.tile(tf.get_variable('sent_init_state_bw_c',
                                                    shape=[1, self.st_hidden_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
                                      multiples=[tf.shape(self.imgs_vec)[0], 1])
            sent_caps_rnn_outputs, _ = bidirectional_rnn(
                cell_fw=cell_fw_c,
                cell_bw=cell_bw_c,
                inputs=self.imgs_vec,
                input_lengths= self.sequence_len_capdoc,
                initial_state_fw=init_state_fw_c,
                initial_state_bw=init_state_bw_c,
                scope=scope
                )
            self.sent_text = tf.tanh(sent_text_rnn_outputs)    # (batch, text_size, st_dim*2)
            self.sent_caps = tf.tanh(sent_caps_rnn_outputs)    # (batch, caps_size, st_dim*2)
            
    def classifier(self):
        with tf.variable_scope('classifier'):
            # training weight for scoring similarity using MLP
#             W_simi = tf.get_variable('W_simi', [self.st_hidden_dim*2, 40], tf.float32,
#                                       initializer = tf.contrib.layers.xavier_initializer(uniform=False))
#             
#             U_simi = tf.get_variable('U_simi', [self.st_hidden_dim*2, 40], tf.float32, 
#                                      initializer = tf.contrib.layers.xavier_initializer(uniform=False))
#             
#             V_simi = tf.get_variable('V_simi', [40,1], tf.float32, initializer = tf.contrib.layers.xavier_initializer(uniform=False))
#             b_simi = tf.get_variable('b_simi', [40], tf.float32, initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            # conv window for post-sco
#             filter = tf.get_variable('filter', [self.win_size,1,1,1],tf.float32,
#                                      initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            
            # training weight for updating mechanism
            W_mid = tf.get_variable('W_mid', [self.st_hidden_dim*4,self.st_hidden_dim*2], tf.float32, 
                                    initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            
            b_mid = tf.get_variable('b_mid', [self.st_hidden_dim*2], tf.float32,
                                     initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            context = self.sent_text
            img_seq = tf.transpose(self.sent_caps, [1,0,2])     # (caps_size, batch, st_dim*2)
            
            # decoder
            out = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True) 
            time = tf.constant(0, tf.int32)
            def cond(time, n, context, out_l):
                return  time < n
            def body(time, n,context, out_l):
                # 1. pre-sco with inner product
                simi = pre_similarity(context, img_seq[time], self.win_size)  #(batch, text_size)

                # 2. pre-sco with MLP
#                 simi = pre_simil_att(context, self.img_seq[time], self.win_size,
#                                     W_simi, U_simi,V_simi,b_simi)
                # 3. post-sco
#                 simi = post_similarity(context, self.img_seq[time], filter) 

                simi_mask = Mask(simi, self.sequence_len_text,tf.reduce_min(simi))
                # update the sentence representation
                context = update_acp(context, img_seq[time], simi, W_mid, b_mid) 
                # concatenate the similarities of each recurrence    
                out_l = out_l.write(time, simi_mask)
                return time+1, n, context, out_l
            _,_,_,output = tf.while_loop(cond, body, [time, tf.shape(self.sent_caps)[1], context, out])
            output = output.stack()
            self.logits = tf.transpose(output, [1, 0, 2])      # (batch, caps_size, text_size)
            
            
    def train(self):     
        # using one hot encode the label; 
        # using cross entropy calculate loss;
        # using Adam 0ptimizate model_withcap.
        labels = tf.one_hot(self.labels, depth=tf.cast(tf.shape(self.text)[1], dtype = tf.int32), axis = -1)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits))
        train_op = tf.train.AdamOptimizer(self.learning_rate, name='Adam').minimize(loss)    
        return  train_op, loss
                  
    def accury(self):
        pri_index = tf.argmax(self.logits, axis = -1)
        mask_pri = Mask(pri_index, self.sequence_len_capdoc,0)
        bias = tf.subtract(mask_pri, self.labels)
        incorrect_num = tf.count_nonzero(bias, axis = -1)
        # err_rate
        incorect_rate_hard = tf.divide(incorrect_num,self.sequence_len_capdoc)
        # err_offset
        incorect_rate_soft = tf.divide(tf.reduce_sum(tf.abs(bias), axis = -1),self.sequence_len_capdoc)
        return incorect_rate_hard, incorect_rate_soft, bias, mask_pri
            
    def get_feed_dict(self, text_dir, captions_dir, img_dir, filenames, sen_maxlen, emb_size):
        text, _, imgs, la, sequence_len_sent, _, sequence_len_text, sequence_len_capdoc = batch_normalize(text_dir, captions_dir, img_dir,
                                                                                                          filenames, sen_maxlen, emb_size)
      
        fd = {
            self.text: text,
            self.imgs: imgs,
            self.labels: la,
            self.sequence_len_sent: sequence_len_sent,
            self.sequence_len_text: sequence_len_text, 
            self.sequence_len_capdoc: sequence_len_capdoc
            }
        return fd



