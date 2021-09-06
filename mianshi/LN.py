import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(1234)

x1 = tf.random.normal((4, 2, 3), mean=10, stddev=10.0)
tf.print(x1)

# tensorflow 实现
x2 = tf.keras.layers.LayerNormalization()(x1)
tf.print(x2)

# 手动实现
x_mean = tf.reduce_mean(x1, axis=-1)
tf.print(x_mean)
x_mean = tf.expand_dims(x_mean, -1)

x_std = tf.math.reduce_std(x1, axis=-1)
tf.print(x_std)
x_std = tf.expand_dims(x_std, -1)

x3 = (x1 - x_mean)/x_std
tf.print(x3)


