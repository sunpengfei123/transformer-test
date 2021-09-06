# 1. load data
# 2. preprocess data -> dataset
# 3. tools
# 3.1 generate position embedding
# 3.2 create mask(a. padding,b decoder)
# 3.3 scaled_dot_product_attention
# 4. builds model
# 4.1 Multi_head Attention
# 4.2 EncoderLayer
# 4.3 DecoderLayer
# 4.4 EncoderModel
# 4.5 DecoderModel
# 4.6 Transformer
# 5. optimizer & loss
# 6. train_step -> train
# 7. Evaluate & Visualization
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time


#  1. load data
examples, info = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)# as_supervised ：若为 True，
# 则根据数据集的特性，将数据集中的每行元素整理为有监督的二元组 (input, label) （即 “数据 + 标签”）形式，否则数据集中的每行元素为包含所有特征的字典。

train_examples, val_examples = examples['train'], examples['validation']
print(info)

for pt, en in train_examples.take(5):
    print(pt.numpy())
    print(en.numpy())
    print()


# 2. preprocess data -> dataset

# 该函数的功能就是把我们的文本做成类似字典的结构，既每个字都有对应的唯一数字
en_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples),
    target_vocab_size=2**13,
    )  # 标记器，是一个词典映射表

pt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples),
    target_vocab_size=2**13,
    )

# 数据预处理（文本id化，过滤不符合规则的句子）

buffer_size = 20000
batch_size = 64
max_length = 40


def encode_to_subword(pt_sentence, en_sentence):
    pt_sequence = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(pt_sentence.numpy()) + [pt_tokenizer.vocab_size+1]
    en_sequence = [en_tokenizer.vocab_size] + en_tokenizer.encode(en_sentence.numpy()) + [pt_tokenizer.vocab_size+1]
    return pt_sequence, en_sequence


def filter_by_max_length(pt, en):
    """过滤掉长句子"""
    return tf.logical_and(tf.size(pt) <= max_length,
                          tf.size(en) <= max_length)


def tf_encode_to_subwords(pt_sentence, en_sentence):
    """对函数进行封装"""
    return tf.py_function(encode_to_subword,
                          [pt_sentence, en_sentence],  # 由一个或者是几个Tensor组成的list 函数encode_to_subword的输入数据
                          [tf.int64, tf.int64]  # 由tensorflow的数据类型组成的list或者是tuple，要与func函数的返回值是匹配的，
                          # 如果是空的话则代表func函数无返回值，即None。
                          )


train_dataset = train_examples.map(tf_encode_to_subwords)
train_dataset = train_dataset.filter(filter_by_max_length)  # filter接收一个函数Func并将该函数作用于dataset的每个元素，
# 根据返回值True或False保留或丢弃该元素，True保留该元素，False丢弃该元素
train_dataset = train_dataset.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=([-1], [-1]))  # 数据分为两个维度，
# 每个维度都扩展到每个batch中句子最高的维度值，例如最长的句子长度作为维度
valid_dataset = val_examples.map(tf_encode_to_subwords)
valid_dataset = valid_dataset.filter(filter_by_max_length)
valid_dataset = valid_dataset.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=([-1], [-1]))

for pt_batch, en_batch in valid_dataset.take(5):
    print(pt_batch.shape, en_batch.shape)


# 3.1 generate position embedding
# 位置编码
# 位置embedding
#
# PE(pos,2i) = sin(pos/10000^(2i/d_model))
# PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
# pos:[sentence_length,1]
# i.shape [1,d_model]
# result.shape:[sentence_length,d_model]

def get_angles(pos, i, d_model):
    # d_model 是embedding的大小
    angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(d_model))
    return pos*angle_rates


def get_position_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # sines.shape:[sentence_length,d_model/2]
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    # [sentence_length,d_model]
    position_embedding = np.concatenate([sines, cosines], axis=-1)
    # [1, sentence_length, d_model]
    position_embedding = position_embedding[np.newaxis, ...]  # 三个点表示遍历某个维度
    return tf.cast(position_embedding, dtype=tf.float32)


position_embedding = get_position_embedding(50, 512)
print(position_embedding.shape)


def plot_position_embedding(position_embedding):
    # 绘制位置编码
    plt.pcolormesh(position_embedding[0], cmap='RdBu')  # [50*512]
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


plot_position_embedding(position_embedding)


# 3.2 create mask(a. padding,b decoder)

# 1. padding mask: loss计算时没必要加上padding
# 2. look ahead mask: decode只能和之前的词语发生关系，不能看到后面的词语
def create_padding_mask(batch_data):
    # batch_data.shape: [batch_size, seq_len]
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


x = tf.constant([[7, 6, 2, 2, 0], [1, 2, 2, 0, 0], [0, 0, 0, 1, 1]])  # 注意没有取反，1表示需要被遮掩的部分
print(create_padding_mask(x))


# attention_weights.shape: [3, 3]
# [[1,2,3],[4,5,6],[7,8,9]] 1表示第一个单词和第一个单词的attention,2表示第一个单词和第二个单词的attention
# [[1,0,0],[4,5,0],[7,8,9]] 创造成一个上三角被mask
def create_look_ahead_mask(size):
    mask = 1-tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # 注意没有取反，1表示需要被遮掩的部分
    return mask  # [seq_len, seq_len]


print(create_look_ahead_mask(3))


# 3.3 scaled_dot_product_attention
# 缩放点积注意力
def scaled_dot_product_attention(q, k, v, mask):
    """

    :param q:  shape = (..., seq_len_q, depth)
    :param k: shape = (..., seq_len_k, depth)
    :param v: shape = (..., seq_len_v, depth_v)
    - seq_len_k = seq_len_v
    :param mask: shape = (..., seq_len_q, seq_len_k)  点积
    :return:
    output: weighted sum
    attention_weights: weights of attention
    """

    # shape == (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1],  tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)
    if mask is not None:
        # 负的10九次方比较小，会使得需要掩盖的数据在softmax的时候趋近0
        scaled_attention_logits += (mask*-1e9)
    # shape == (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # shape = (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def print_scaled_dot_attention(q, k, v):
    temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
    print("Attention weights are:")
    print(temp_att)
    print("Outputs are:")
    print(temp_out)


# 测试
temp_k = tf.constant([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=tf.float32) # (4,3)
temp_v = tf.constant([[1, 0], [10, 0], [100, 5], [1000, 6]], dtype=tf.float32) #(4,2)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  #(1,3)
np.set_printoptions(suppress=True)  # 使得小数结果压缩
print_scaled_dot_attention(temp_q, temp_k,temp_v)
temp_q2 = tf.constant([[0, 0, 10]], dtype=tf.float32)  # temp_k最后一列平分权重,根据此结果平分temp_v的最后两行
print_scaled_dot_attention(temp_q2, temp_k, temp_v)
temp_q3 = tf.constant([[0, 10, 0], [0, 0, 10]], dtype=tf.float32)
print_scaled_dot_attention(temp_q3, temp_k, temp_v)  # 拼接


# 4. builds model
# 4.1 Multi_head Attention
# 多头注意力实现
#     '''
#     理论上
#     x->Wq0->q0
#     x->Wk0->k0
#     x->Wv0->v0
#     实战中
#     q->Wq0->q0
#     k->Wk0->k0
#     v->Wv0->v0
#     技巧
#     q->Wq->Q->split->q0,q1,...
#     '''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # // 表示整数除法

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        分拆最后一个维度到 (num_heads, depth)
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        :param x:
        :param batch_size:
        :return:
        """
        x  = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, att = temp_mha(y, k=y, q=y, mask=None)
print(out.shape, att.shape)


# 4.2 EncoderLayer
def feed_forward_network(d_model, diff):
    # 前馈神经网络
    # diff: dim of feed network
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])



sample_ffn = feed_forward_network(512,2048)
sample_ffn(tf.random.uniform((64,50,512))).shape


class EncoderLayer(tf.keras.layers.Layer):
    """
    block:
    x->self.attention->add&normalize&dropout->feed_forward->add&normalize&dropout
    """
    def __init__(self, d_model, num_heads, diff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads=num_heads)
        self.ffn = feed_forward_network(d_model, diff=diff)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)


    def call(self, x, training, encoder_padding_mask):
        # x.shape:(batch_size,seq_len,dim=dmodel)
        # attn_shape:(batch_size,seq_len,d_model)
        attn_output, _ = self.mha(x, x, x, encoder_padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x+attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output(batch_size,seq_len,d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # out2.shape:(batch_size,seq_len,d_model)
        out2 = self.layer_norm2(out1+ffn_output)
        return out2


sample_encoder_layer = EncoderLayer(512,8,2048)
sample_input =tf.random.uniform((64,50,512))
sample_output = sample_encoder_layer(sample_input,False,None)
print(sample_output.shape)


class DecoderLayer(tf.keras.layers.Layer):
    """
    x->self.Attention->add&norm&dropout->out1
    out1,encoding_outputs->self.attention_>add&norm&dropput->out2
    out2->ffn->add&norm&dropout->out3
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, decoder_mask, encoder_decoder_padding_mask):
        """
        decoder_mask是decoder_padding_mask和look_ahead_mask的结合体
        :param x:
        :param enc_output:
        :param training:
        :param decoder_mask:
        :param encoder_decoder_padding_mask:
        :return:
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, encoder_decoder_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, attn_weights1, attn_weights2 = sample_decoder_layer(
    tf.random.uniform((64, 60, 512)), sample_output,
    False, None, None)

print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
print(attn_weights1.shape)
print(attn_weights2.shape)


class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        # shape:(1,max_len,d_model)
        self.position_embedding = get_position_embedding(max_length, self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]

    def call(self, x, training, encoder_padding_mask):
        # x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1]
        # assert input_seq_len <= self.max_length
        tf.debugging.assert_less_equal(input_seq_len, self.max_length, "input_seq_len should be less or equal to self.max_length! ")
        # (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # 缩放，x由（0，1）->（0，d_model）
        x += self.position_embedding[:, :input_seq_len, :]
        x = self.dropout(x, training)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)
        # x.shape:(batch_size,input_seq_len,d_model)
        return x


sample_encoder_model =EncoderModel(2,8500,40,512,8,2048)
sample_encoder_model_input = tf.random.uniform((64,37))
sample_enocder_model_output = sample_encoder_model(sample_encoder_model_input,False,encoder_padding_mask=None)
print(sample_enocder_model_output.shape)

class DecoderModel(tf.keras.layers.Layer):
    def __init__(self, num_layers, target_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.position_embedding = get_position_embedding(max_length, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]

    def call(self, x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        # assert output_seq_len<= self.max_length
        tf.debugging.assert_less_equal(output_seq_len, self.max_length, "output_seq_len should less or equal to self.max_length! ")
        attention_weights = {}
        # x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :output_seq_len, :]
        x = self.dropout(x, training)

        for i in range(self.num_layers):
            x, att1, att2 = self.decoder_layers[i](x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask)
            attention_weights['decoder_layer{}_att1'.format(i + 1)] = att1
            attention_weights['decoder_layer{}_att2'.format(i + 1)] = att2
            # x.shape: (batch_size, output_seq_len, d_model)
        return x, attention_weights


sample_decoder_model = DecoderModel(2,8000,40,512,8,2048)
sample_decoder_model_input = tf.random.uniform((64,35))
sample_decoder_model_output,sample_decoder_model_attn = sample_decoder_model(sample_decoder_model_input,sample_enocder_model_output,training=False,decoder_mask=None,encoder_decoder_padding_mask=None)
print(sample_decoder_model_output.shape)
for key in sample_decoder_model_attn:
    print(sample_decoder_model_attn[key].shape)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder_model = EncoderModel(num_layers, input_vocab_size, max_length, d_model, num_heads, dff, rate)
        self.decoder_model = DecoderModel(num_layers, target_vocab_size, max_length, d_model, num_heads, dff, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, encoding_padding_mask, decoder_mask, encoder_decoder_padding_mask):
        # (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(inp, training, encoding_padding_mask)
        # decoding_outputs:(batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weights = self.decoder_model(tar, encoding_outputs, training,
                                                                 decoder_mask, encoder_decoder_padding_mask)
        # batch_size, output_seq_len, target_vocab_size
        predictions = self.final_layer(decoding_outputs)
        return predictions, attention_weights



# sample_transformer = Transformer(2,8500,8000,40,512,8,2048,rate=0.1)
# temp_input =tf.random.uniform((64,26))
# temp_target = tf.random.uniform((64,31))
# predictions,attention_weights = sample_transformer(temp_input,temp_target,training=False,encoding_padding_mask=None,decoder_mask=None,encoder_decoder_padding_mask=None)
# print(predictions.shape)
# for key in attention_weights:
#     print(attention_weights[key].shape)


# 模型训练

# 1. initial model
# 2. define loss optimizer,learning rate schedule
# 3. train_step
# 4. train process

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tf.math.maximum(pt_tokenizer.vocab_size, en_tokenizer.vocab_siz) + 2  # 要加上start和end
target_vocab_size = input_vocab_size
print('input_vocab_size',input_vocab_size)
print('target_vocab_size',target_vocab_size)
dropout_rate = 0.1

transformer = Transformer(num_layers, input_vocab_size, target_vocab_size, max_length,
                          d_model, num_heads, dff, dropout_rate)  # 初始化


# lrate = (d_model **-0.5) * min(step_num **-0.5,step_num*warm_up_steps**-1.5) 先增后减的学习率
class CustomizedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warm_up_steps=4000):
        super(CustomizedSchedule, self).__init__()
        self.d_model = tf.cast(d_model,tf.float32)
        self.warm_up_steps = warm_up_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps**(-1.5))
        arg3 = tf.math.rsqrt(self.d_model)
        return arg3 * tf.math.minimum(arg1, arg2)


learning_rate = CustomizedSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

temp_learning_rate_schedule = CustomizedSchedule(d_model)
# 0-40000步
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("train_step")
plt.show()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, predict):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # padding上的值为0
    loss_ = loss_object(real, predict)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def create_mask(inp, tar):
    """
    Encoder:
        - encoder_padding_mask (self attention of Encoderlayer)
    Decoder:
        - look_ahead_mask (self attention of DecoderLayer)
        - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
        - decoder_padding_mask(self attention of DecoderLayer)
    :param inp:
    :param tar:
    :return:
    """
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)
    decoder_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)  # 只能传入一个mask，所以要做与操作
    #     print(encoder_padding_mask.shape)
    #     print(encoder_decoder_padding_mask.shape)
    #     print(look_ahead_mask.shape)
    #     print(decoder_padding_mask.shape)
    #     print(decoder_mask.shape) 这里有广播机制的作用
    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask


# 测试代码
temp_inp, temp_tar = iter(train_dataset.take(1)).next()
print(temp_inp.shape)
print(temp_tar.shape)
create_mask(temp_inp, temp_tar)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_mask(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)

    grads = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(grads, transformer.trainable_variables))  # 应用三步走，损失函数，求梯度，进行梯度下降操作
    train_loss(loss)  # 累计平均值
    train_accuracy(tar_real, predictions)


epochs = 10

for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 100 == 0:
            print("Epoch {} Batch {} Accuracy {}".format(epoch+1, batch, train_loss.result(), train_accuracy.result()))

    print('Epoch {} Loss {:.4f} Accuarcy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time take for 1 epoch:{} secs\n'.format(time.time() - start))

"""
eq:ABCD->EFGH
Train:ABCD,EFG ->FGH
Eval:ABCD->E
    ABCD,E ->F
    ABCD,EF ->G
"""


def evalute(inp_sentence):
    input_id_sentence = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size + 1]
    encoder_input = tf.expand_dims(input_id_sentence, 0)  # (1,input_sentence_length)
    decoder_input = tf.expand_dims([en_tokenizer.vocab_size], 0)  # (1,1)
    for i in range(max_length):
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_mask(encoder_input, decoder_input)
        # (batch_size,output_target_len,target_vocab_size)
        predictions, attention_weights = transformer(encoder_input, decoder_input, False, encoder_padding_mask,
                                                     decoder_mask,
                                                     encoder_decoder_padding_mask)
        predictions = predictions[:, -1, :]  # 单步
        predictions_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)  # 预测概率最大的值
        if tf.equal(predictions_id, en_tokenizer.vocab_size + 1):
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, [predictions_id]],
                                  axis=-1)
    return tf.squeeze(decoder_input, axis=0), attention_weights


def plot_encoder_decoder_attention(attention, input_sentence, result, layer_name):
    # attention_weights存的是字典，layer_name是key
    fig = plt.figure(figsize=(16, 8))
    input_id_sentence = pt_tokenizer.encode(input_sentence)
    # attention[layer_name].shape (num_heads,tar_len,input_len)
    attention = tf.squeeze(attention[layer_name], axis=0)
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)  # 两行四列
        ax.matshow(attention[head][:-1, :])  # 绘图
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(input_sentence) + 2))  # 设置锚点数目，start_id end_id
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result) - 1.5, -0.5)
        # 设置锚点对应的单词
        ax.set_xticklabels(
            ['<start>'] + [pt_tokenizer.decode([i]) for i in input_id_sentence] + ['<end>'],
            fontdict=fontdict,
            rotation=90,
        )
        ax.set_yticklabels(
            [en_tokenizer.decode([i]) for i in result if i < en_tokenizer.vocab_size]  # 把start_id和end_id排除掉
            , fontdict=fontdict)
        ax.set_xlabel('Head{}'.format(head + 1))  # 设置总的label
    plt.tight_layout()  # 自适应调整间距
    plt.show()


def translate(input_sentence, layer_name=''):
    result, attention_weights = evalute(input_sentence)
    predicted_sentence = en_tokenizer.decode([i for i in result if i < en_tokenizer.vocab_size])  # 防止无词出错
    print("Input: {}".format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))
    if layer_name:
        plot_encoder_decoder_attention(attention_weights, input_sentence, result, layer_name)


translate('frio', layer_name='decoder_layer4_att2')  # layername来源为自己设定decoder_layer中