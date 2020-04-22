from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class Net(object):

    def __init__(self, is_training=True):

        self.mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

        self.learning_rate = 1e-4
        self.step_numbers = 50
        self.batch_size = 256
        self.dropout = 0.85
        self.n_batch = self.mnist.train.num_examples // self.batch_size
        self.model_path = './model.ckpt'
        self.is_training = is_training

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

        self.pred = self.liner_network(self.x)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        self.correct_pred = tf.equal(tf.arg_max(
            self.pred, 1), tf.arg_max(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver()

    def liner_network(self, inputs):

        net = self.conv2d(inputs, filters=128)

        net = self.conv2d(net, filters=64)

        net = self.max_pool2d(net, k=3)

        net = tf.contrib.layers.flatten(net)

        weights_1 = tf.Variable(tf.random_normal(
            [64*14*14, 128], stddev=0.2))

        biases_1 = tf.Variable(
            tf.random.normal(shape=[128])
        )

        logits = tf.add(tf.matmul(net, weights_1), biases_1)

        weights_2 = tf.Variable(tf.random_normal(
            [128, 10], stddev=0.2))

        biases_2 = tf.Variable(
            tf.random.normal(shape=[10])
        )

        logits = tf.add(tf.matmul(logits, weights_2), biases_2)

        if self.is_training:
            logits = tf.nn.dropout(logits, rate=1-self.dropout)

        predictions = tf.nn.softmax(logits)

        return predictions

    def network(self, inputs):

        net = self.conv2d(inputs, filters=128)
        net = self.conv2d(net, filters=128)

        net = self.max_pool2d(net, k=3, strides=2)

        net = self.conv2d(net, filters=128, strides=2)
        net = self.conv2d(net, filters=64)

        net = self.max_pool2d(net, k=3)

        net = self.conv2d(net, filters=64)

        net = self.max_pool2d(net, k=3)

        net = self.conv2d(net, filters=32)

        net = self.max_pool2d(net, k=3)

        net = tf.contrib.layers.flatten(net)

        weights = tf.Variable(tf.random_normal(
            [32, 10], stddev=0.2))

        biases = tf.Variable(
            tf.random.normal(shape=[10])
        )

        logits = tf.add(tf.matmul(net, weights), biases)

        if self.is_training:
            logits = tf.nn.dropout(logits, rate=1-self.dropout)

        predictions = tf.nn.softmax(logits)

        return predictions

    def conv2d(self, x, filters, k_size=3, strides=1, padding='SAME', dilation=[1, 1]):

        return tf.layers.conv2d(x, filters, kernel_size=[k_size, k_size], strides=[strides, strides],
                                dilation_rate=dilation, padding=padding, activation=tf.nn.relu,
                                use_bias=True)

    def max_pool2d(self, x, k, strides=2, padding='SAME'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=k, strides=strides, padding=padding)

    def train_net(self):

        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.getcwd())

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for i in range(self.step_numbers):

                for _ in range(self.n_batch):

                    batch_x, batch_y = self.mnist.train.next_batch(
                        self.batch_size)

                    feed_dict = {self.x: np.expand_dims(batch_x.reshape(-1, 28, 28), axis=-1),
                                 self.y: batch_y}

                    _, loss = sess.run([self.train_step, self.loss], feed_dict)

                    print('step:{} loss:{}'.format(i, loss), end='\r')

                test_dict = {self.x: np.expand_dims(self.mnist.test.images[:1000].reshape(-1, 28, 28), axis=-1),
                             self.y: self.mnist.test.labels[:1000]}

                acc_test = sess.run(self.accuracy, feed_dict=test_dict)

                print('{}/{} acc:{}'.format(i, self.step_numbers, acc_test))

                self.saver.save(sess, self.model_path)


if __name__ == "__main__":
    net_obj = Net()

    net_obj.train_net()
