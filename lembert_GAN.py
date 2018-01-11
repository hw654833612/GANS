# -*- coding: utf-8 -*-
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
@author: lembert
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


""" ========================================================================== """
""" 定义 generator、discriminator、get_noise """
def generator(noise_z):    # noise >> input
    hidden_layer = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    generated_outputs = tf.sigmoid(tf.matmul(hidden_layer, G_W2) + G_b2)
    return generated_outputs
def discriminator(inputs):   # input >> 1
    hidden_layer = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    discrimination = tf.sigmoid(tf.matmul(hidden_layer, D_W2) + D_b2)
    return discrimination
def get_noise(batch_size):
    return np.random.normal(size=(batch_size, n_noise))





""" ========================================================================== """
""" 开始定义网络结构 """
n_input = 784
n_hidden = 256
n_noise = 128  
learning_rate = 0.0002


G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))



X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

G = generator(Z)                           # [None, n_noise] >> [None, n_input]
D_gene = discriminator(G)                  # [None, n_input] >> [None, 1]
D_real = discriminator(X)                  # [None, n_input] >> [None, 1]
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene)) 
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)


""" ========================================================================== """
""" 执行tensorflow定义的结构"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0
Sample_all = []
for epoch in range(1000):
    if epoch<25:
        CRITIC_NUM = 5
    else:
        CRITIC_NUM = 3
    for i in range(total_batch):
        for _ in range(CRITIC_NUM):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            noise = get_noise(batch_size)
            _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})
    print 'Epoch:', '%04d' % (epoch + 1), 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G)
    noise = get_noise(100)
    samples = sess.run(G, feed_dict={Z: noise})
    Sample_all.append(samples)
    
    fig, ax = plt.subplots(10, 10, figsize=(10, 10)) # 可视化
    for i1 in range(10):
        for i2 in range(10):
            ax[i1][i2].set_axis_off()
            ax[i1][i2].imshow(np.reshape(samples[i1*10+i2], (28, 28)), cmap="gray")
    plt.savefig('Figure/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)




