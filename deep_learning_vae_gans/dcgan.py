import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=6*6*128, activation=None)
        x = tf.nn.relu(x)
        x = tf.reshape(x, (-1, 6, 6, 128))

        x = tf.layers.conv2d_transpose(x, 128, kernel_size=[4, 4], strides=[2, 2],
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                       bias_initializer=tf.zeros_initializer())
        x = tf.nn.relu(x)

        x = tf.layers.conv2d_transpose(x, 1, kernel_size=[2, 2], strides=[2, 2],
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                       bias_initializer=tf.zeros_initializer())
        x = tf.nn.sigmoid(x)
        return x

def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, kernel_size=[5, 5], filters=2, strides=[2, 2], 
                             padding="valid", activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                             bias_initializer=tf.zeros_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = tf.layers.conv2d(x, kernel_size=[5, 5], filters=64, strides=[2, 2], 
                             padding="valid", activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                             bias_initializer=tf.zeros_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=2, activation=None)
        return x

if __name__ == "__main__":
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    noise_dim = 200
    num_steps = 20000
    batch_size = 32

    real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    z = tf.placeholder(tf.float32, shape=[None, noise_dim])
    disc_target = tf.placeholder(tf.int32, shape=[None])
    gen_target = tf.placeholder(tf.int32, shape=[None])
    
    gen_sample = generator(z)
    disc_real = discriminator(real_image_input)
    disc_fake = discriminator(gen_sample, reuse=True)

    disc_concat = tf.concat([disc_real, disc_fake], axis=0)
    stacked_gan = discriminator(gen_sample, reuse=True)

    disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_concat, labels=disc_target))
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=stacked_gan, labels=gen_target))

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    with tf.Session() as sess:
        tf.set_random_seed(42)
        np.random.seed(42)
        sess.run(tf.global_variables_initializer())

        for i in range(1, num_steps+1):
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([batch_x.shape[0], 28, 28, 1])

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])

            batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
            batch_gen_y = np.ones([batch_size])

            feed_dict = {real_image_input: batch_x, z: noise,
                         disc_target: batch_disc_y, gen_target: batch_gen_y}

            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)

            if i % 100 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        f, a = plt.subplots(4, 10, figsize=[10, 4])
        for i in range(10):
            noise = np.random.uniform(-1.0, 1.0, size=[4, noise_dim])
            g = sess.run(gen_sample, feed_dict={z: noise})
            for j in range(4):
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=[28, 28, 3])
                a[j][i].imshow(img)

        f.show()
        plt.draw()
        plt.waitforbuttonpress()
