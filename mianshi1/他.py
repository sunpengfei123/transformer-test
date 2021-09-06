import tensorflow as tf
from keras import layers


class sda(layers.Layer):
    def __init__(self):
        super(sda, self).__init__()

    def call(self, q, k, v, mask):
        mt = tf.matmul(q, k, transpose_b=True)
        dk = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        logits = mt/dk
        if not mask == None:
            logits+=mask

        we = tf.nn.softmax(logits)

        out = tf.matmul(we,v)

        return out, we



class MHA(layers.Layer):
    def __init__(self, num_heads, d_model, batch_size):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model//num_heads
        self.bs = batch_size
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.out = layers.Dense(d_model)

    def split(self,x):
        x = tf.reshape(x, (self.bs, -1, self.num_heads, self.depth))

        x = tf.transpose(x, perm=[0,2,1,3])

        return x

    def call(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        out1, w1 = sfa(q, k, v, mask)
        out1 = tf.transpose(out1, perm=[0,2,1,3])
        out1 = tf.reshape(out1, (self.bs, -1, self.d_model))

        out1 = self.out(out1)

        return out1, w1