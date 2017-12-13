import tensorflow as tf

n_input = 28 * 28
n_l1 = 256
n_l2 = 128
n_l3 = 64
n_l4 = 26
skip_reg = 0.00005
vanilla_reg = 0.0005

class skip_network:
    def __init__(self, scope, lr):
        self.scope = scope
        with tf.variable_scope(scope):
            self.n_input = n_input
            self.n_l1 = n_l1
            self.n_l2 = n_l2
            self.n_l3 = n_l3
            self.n_l4 = n_l4
            self.reg = skip_reg

            self.raw_input = tf.placeholder(tf.float32, [None, 28, 28], name='input')
            self.input = tf.reshape(self.raw_input, [-1, 784])

            self.W1 = tf.get_variable('W1', [self.n_input, self.n_l1], initializer=tf.contrib.layers.xavier_initializer())
            self.W2 = tf.get_variable('W2', [self.n_l1, self.n_l2], initializer=tf.contrib.layers.xavier_initializer())
            self.W3 = tf.get_variable('W3', [self.n_l2, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.W4 = tf.get_variable('W4', [self.n_l3, self.n_l4], initializer=tf.contrib.layers.xavier_initializer())
            self.W0_2 = tf.get_variable('W0_2', [self.n_input, self.n_l2], initializer=tf.contrib.layers.xavier_initializer())
            self.W1_3 = tf.get_variable('W1_3', [self.n_l1, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.W0_3 = tf.get_variable('W0_3', [self.n_input, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.W2_4 = tf.get_variable('W2_4', [self.n_l2, self.n_l4], initializer=tf.contrib.layers.xavier_initializer())
            self.W1_4 = tf.get_variable('W1_4', [self.n_l1, self.n_l4], initializer=tf.contrib.layers.xavier_initializer())
            self.W0_4 = tf.get_variable('W0_4', [self.n_input, self.n_l4], initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable('b1', [self.n_l1], initializer=tf.constant_initializer(0.01))
            self.b2 = tf.get_variable('b2', [self.n_l2], initializer=tf.constant_initializer(0.01))
            self.b3 = tf.get_variable('b3', [self.n_l3], initializer=tf.constant_initializer(0.01))
            self.b4 = tf.get_variable('b4', [self.n_l4], initializer=tf.constant_initializer(0.00))

            self.layer1 = tf.nn.relu(tf.matmul(self.input, self.W1) + self.b1)
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + tf.matmul(self.input, self.W0_2) + self.b2)
            self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.W3) + tf.matmul(self.layer1, self.W1_3) + tf.matmul(self.input, self.W0_3) + self.b3)
            self.layer4 = tf.matmul(self.layer3, self.W4) + tf.matmul(self.layer2, self.W2_4) + tf.matmul(self.layer1, self.W1_4) + tf.matmul(self.input, self.W0_4) + self.b4

            self.predictions = tf.argmax(self.layer4, axis=1) + 1

            self.labels = tf.placeholder(tf.int32, [None], name='labels')
            self.labels_modified = self.labels - 1
            self.labels_onehot = tf.one_hot(indices=self.labels_modified, depth=26)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_onehot, logits=self.layer4)) + \
                        self.reg * (tf.reduce_sum(self.W1 * self.W1) + tf.reduce_sum(self.W2 * self.W2) + tf.reduce_sum(self.W3 * self.W3) + tf.reduce_sum(self.W4 * self.W4) + \
                        tf.reduce_sum(self.W0_2 * self.W0_2) + \
                        tf.reduce_sum(self.W1_3 * self.W1_3) + tf.reduce_sum(self.W0_3 * self.W0_3) + \
                        tf.reduce_sum(self.W2_4 * self.W2_4) + tf.reduce_sum(self.W1_4 * self.W1_4) + tf.reduce_sum(self.W0_4 * self.W0_4))

            self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

class vanilla_network:
    def __init__(self, scope, lr):
        self.scope = scope
        with tf.variable_scope(scope):
            self.n_input = n_input
            self.n_l1 = n_l1
            self.n_l2 = n_l2
            self.n_l3 = n_l3
            self.n_l4 = n_l4
            self.reg = vanilla_reg

            self.raw_input = tf.placeholder(tf.float32, [None, 28, 28], name='input')
            self.input = tf.reshape(self.raw_input, [-1, 784])

            self.W1 = tf.get_variable('W1', [self.n_input, self.n_l1], initializer=tf.contrib.layers.xavier_initializer())
            self.W2 = tf.get_variable('W2', [self.n_l1, self.n_l2], initializer=tf.contrib.layers.xavier_initializer())
            self.W3 = tf.get_variable('W3', [self.n_l2, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.W4 = tf.get_variable('W4', [self.n_l3, self.n_l4], initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable('b1', [self.n_l1], initializer=tf.constant_initializer(0.01))
            self.b2 = tf.get_variable('b2', [self.n_l2], initializer=tf.constant_initializer(0.01))
            self.b3 = tf.get_variable('b3', [self.n_l3], initializer=tf.constant_initializer(0.01))
            self.b4 = tf.get_variable('b4', [self.n_l4], initializer=tf.constant_initializer(0.00))

            self.layer1 = tf.nn.relu(tf.matmul(self.input, self.W1) + self.b1)
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
            self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.W3) + self.b3)
            self.layer4 = tf.matmul(self.layer3, self.W4) + self.b4

            self.predictions = tf.argmax(self.layer4, axis=1) + 1

            self.labels = tf.placeholder(tf.int32, [None], name='labels')
            self.labels_modified = self.labels - 1
            self.labels_onehot = tf.one_hot(indices=self.labels_modified, depth=26)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_onehot, logits=self.layer4)) + \
                        self.reg * (tf.reduce_sum(self.W1 * self.W1) + tf.reduce_sum(self.W2 * self.W2) + tf.reduce_sum(self.W3 * self.W3) + tf.reduce_sum(self.W4 * self.W4))

            self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
