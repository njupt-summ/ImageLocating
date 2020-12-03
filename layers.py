import tensorflow as tf
import numpy as np 
try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs, scope=None):
    with tf.variable_scope(scope or 'bi_rnn') as scope:
        (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
#             initial_state_fw=initial_state_fw,
#             initial_state_bw=initial_state_bw,
            dtype=tf.float32,scope=scope)
        outputs = tf.concat((fw_outputs, bw_outputs), axis=2)
        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                  len(fw_state) == len(bw_state)):
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state
            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state


# 先计算相似度，再考虑上下文的影响

def distance(text, img):
    
    # 欧式距离
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(text-img), axis = 1))
    
    #cos 距离
    x1 = tf.sqrt(tf.reduce_sum(tf.square(img),axis = 0))
    x2 = tf.sqrt(tf.reduce_sum(tf.square(text),axis = 1))
    numerator = tf.reduce_sum(tf.multiply(text,img),axis = 1)
    cos_dis = tf.divide(numerator,tf.multiply(x1, x2))
    return cos_dis


def pre_similarity(text, img, win_size):

    simi = tf.matmul(text, tf.reshape(img,shape=[-1,1]))    # 内积表示距离
#     simi = tf.reshape(distance(text, img),shape = [-1,1])
    context_simi = tf.layers.average_pooling1d(tf.expand_dims(simi, axis=0), pool_size=win_size, 
                                               strides=1, padding = 'same')
    return tf.squeeze(context_simi)
#     return tf.reduce_mean(context_simi[0],axis = 1)

# tanh(wq + Ui + b)V
def pre_simi_att(text, img, win_size, W, U, V, b):
#     
    cover = tf.matmul(text, W) + tf.matmul(tf.expand_dims(img, axis = 0), U) +b
    simi = tf.matmul(tf.tanh(cover), V)
    context_simi = tf.layers.average_pooling1d(tf.expand_dims(simi, axis=0), pool_size=win_size, 
                                               strides=1, padding = 'same')
    return tf.squeeze(context_simi)

# 正态分布密度函数生成
def nomal(u,rang):
    x = tf.cast(tf.range(rang), tf.int64)
    sig = tf.cast(rang/2, tf.int64)
    out = tf.cast(tf.exp(-tf.square((x - u)) / (2 * tf.square(sig))),tf.float32)
    return out

def update_context(text, img, alpha, last_img_lo, weight, bias):
  

    # beta和img均为一维
#     beta = tf.multiply(nomal(last_img_lo, tf.shape(text)[0]),alpha)
    beta = nomal(last_img_lo, tf.shape(text)[0])
#     扩展纬度再乘
#     beta_expand = tf.tile(tf.expand_dims(beta, axis=1),[1, tf.shape(img)[0]])
#     img_expand = tf.tile(tf.expand_dims(img, axis=0),[tf.shape(text)[0], 1])
#     txt_add_img = text + tf.multiply(beta_expand, img_expand)



#     txt_add_img = text + tf.multiply(tf.multiply(tf.expand_dims(beta, axis=1), img),text)      # 图片对其他文本的影响，方式2

#     txt_add_img = text + tf.multiply(tf.expand_dims(beta, axis=1),img)      # 图片对其他文本的影响，方式1
#     up_text = tf.tanh(tf.matmul(txt_add_img, weight) + bias)
    txt_add_img = tf.concat([text, tf.multiply(tf.expand_dims(beta, axis=1),img)], axis=1)      # 图片对其他文本的影响，方式4
    up_text = tf.tanh(tf.matmul(txt_add_img,weight)+bias)


#     txt_with_img = tf.concat([txt_add_img[:last_img_lo,:],tf.expand_dims(img,axis = 0), txt_add_img[last_img_lo:,:]], axis = 0) # 插入图片向量
#     img_impact = tf.matmul(tf.multiply(tf.expand_dims(beta, axis=1),img),weight)+bias       # 图片对其他文本的影响， 方式3，                                
#     up_text = tf.tanh(text + tf.tanh(img_impact))
    
    return up_text

# 先计算上下文的影响，再计算相似度
def post_similarity(text, img, filter , W, U, V, b):
    text_expand = tf.expand_dims(tf.expand_dims(text,axis = -1), axis = 0)
    del text
    context = tf.squeeze(tf.nn.conv2d(text_expand, filter, strides=[1,1,1,1], padding='SAME'))
#     simi = tf.squeeze(tf.matmul(context, tf.reshape(img,shape=[-1,1])))
    cover = tf.tanh(tf.matmul(context, W) + tf.matmul(tf.expand_dims(img, axis = 0), U) +b)
    simi = tf.squeeze(tf.matmul(cover, V))
    return simi


# u = tf.constant(2,tf.int64)
# a= tf.constant(20,tf.int64)
# d = nomal(u,a)
# print(d)


