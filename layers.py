import tensorflow as tf
import numpy as np 
try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

def bidirectional_rnn(cell_fw, cell_bw, inputs, input_lengths, initial_state_fw=None, initial_state_bw=None, scope=None):
    with tf.variable_scope(scope or 'bi_rnn') as scope:
        (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=input_lengths,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32,
            scope=scope)
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

# pre-sco with cosine distance or inner product
def pre_similarity(text, img, win_size):
    # text(batch, text_size, st_dim*2)
    # img(batch, st_dim*2)
    img = tf.expand_dims(img, axis = -1)
    simi = tf.matmul(text, img)        #(batch, text_size, 1)
    context_simi = tf.layers.average_pooling1d(simi, pool_size=win_size, 
                                               strides=1, padding = 'same')
    return tf.squeeze(context_simi)

# post-sco with inner product
def post_similarity(text, img):
    # text(batch, text_size, st_dim*2)
    # img(batch, st_dim*2)
    text_expand = tf.expand_dims(text,axis = -1)
    context = tf.squeeze(tf.nn.conv2d(text_expand, filter, strides=[1,1,1,1], padding='SAME'))
    simi = tf.squeeze(tf.matmul(context, tf.expand_dims(img,axis=-1)))
    return simi

# pre-sco with MLP (V*tanh(wq + Ui + b))
def pre_simil_att(text, img, win_size, W, U, V, b):
    # text(batch, text_size, st_dim*2)
    # img(batch, st_dim*2)    
    cover = tf.matmul(text, W) + tf.expand_dims(tf.matmul(img, U), axis = 1) +b
    simi = tf.matmul(tf.tanh(cover), V)
    context_simi = tf.layers.average_pooling1d(simi,pool_size=win_size, 
                                               strides=1, padding = 'same')
    return tf.squeeze(context_simi)

# post-sco with MLP
def post_simil_att(text, img, filter , W, U, V, b):
    # text(batch, text_size, st_dim*2)
    # img(batch, st_dim*2)
    text_expand = tf.expand_dims(text,axis = -1)
    context = tf.squeeze(tf.nn.conv2d(text_expand, filter, strides=[1,1,1,1], padding='SAME'))
    cover = tf.tanh(tf.matmul(context, W) + tf.expand_dims(tf.matmul(img, U), axis = 1) +b)
    simi = tf.squeeze(tf.matmul(cover, V))
    return simi


def update_acp(text, img, alpha, weight, bias):
    # text(batch, text_size, st_dim*2)
    # img(batch, st_dim*2)
    # alpha(batch, text_size)
    txt_add_img = tf.concat([text, 
                             tf.multiply(tf.expand_dims(alpha, axis=-1),tf.expand_dims(img, axis=1))],
                             axis=-1)
    # txt_add_img(batch, text_size, st_dim*4)
    up_text = tf.tanh(tf.matmul(txt_add_img,weight)+bias)
    return up_text

def Mask(ini_value, sequence_lengths, mask_value):
    # ini_value(batch, text_size)
    # sequence_lengths(batch,)
    # mask_value: minimum value
    mark = tf.sequence_mask(sequence_lengths, tf.shape(ini_value)[-1])
    fill = mask_value*tf.ones_like(ini_value)
    return tf.where(mark,ini_value,fill)

