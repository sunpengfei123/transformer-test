import tensorflow as tf
from tensorflow import keras
from keras import layers


def scaled_dot_product_attention(q, k, v, mask):
    qk = tf.matmul(q,k,transpose_b=True)
    dk = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))

    logits = qk/dk

    if mask is not None:
        logits += (mask*(-1e9))

    atten = tf.nn.softmax(logits, axis=-1)
    out = tf.matmul(atten, v)
    return out, atten

class MHA(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        sca, atten = scaled_dot_product_attention(q, k, v, mask)

        sca = tf.transpose(sca, perm=[0,2,1,3])

        sca = tf.reshape(sca, (batch_size, -1, self.d_model))

        out = self.dense(sca)

        return out, atten
