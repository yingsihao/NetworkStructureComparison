n_input = 28 * 28
n_l1 = 512
n_l2 = 256
n_l3 = 128

class skip_network:
    def __init__(self, scope, lr):
        self.scope = scope
        with tf.variable_scope(scope):
            self.n_input = n_input
            self.n_l1 = n_l1
            self.n_l2 = n_l2
            self.n_l3 = n_l3

            self.input = tf.placeholder(tf.float32, [None, n_input], name='input')

            self.W1 = tf.get_variable('W1', [self.n_input, self.n_l1], initializer=tf.contrib.layers.xavier_initializer())
            self.W2 = tf.get_variable('W2', [self.n_l1, self.n_l2], initializer=tf.contrib.layers.xavier_initializer())
            self.W3 = tf.get_variable('W3', [self.n_l2, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.W4 = tf.get_variable('W4', [self.n_input, self.n_l2], initializer=tf.contrib.layers.xavier_initializer())
            self.W5 = tf.get_variable('W5', [self.n_input, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.W6 = tf.get_variable('W6', [self.n_l1, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable('b1', [self.n_l1], initializer=tf.constant_initializer(0.01))
            self.b2 = tf.get_variable('b2', [self.n_l2], initializer=tf.constant_initializer(0.01))
            self.b3 = tf.get_variable('b3', [self.n_l3], initializer=tf.constant_initializer(0.00))

            self.layer1 = tf.nn.relu(tf.matmul(self.input, self.W1) + self.b1)
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + tf.matmul(self.input, self,W4) + self.b2)
            self.layer3 = tf.matmul(self.layer2, self.W3) + tf.matmul(self.layer1, self.W5) + tf.matmul(self.input, self.W6) + self.b3

            self.predictions = tf.argmax(self.layer3, axis=1) + 1

            self.labels = tf.placeholder(tf.int32, [None], name='labels')
            self.labels = self.labels - 1
            self.labels_onehot = tf.one_hot(indices=self.label, depth=26)

            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.labels_onehot, self.layer3))

            self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

class vanilla_network:
    def __init__(self, scope, lr):
        self.scope = scope
        with tf.variable_scope(scope):
            self.n_input = n_input
            self.n_l1 = n_l1
            self.n_l2 = n_l2
            self.n_l3 = n_l3

            self.input = tf.placeholder(tf.float32, [None, n_input], name='input')

            self.W1 = tf.get_variable('W1', [self.n_input, self.n_l1], initializer=tf.contrib.layers.xavier_initializer())
            self.W2 = tf.get_variable('W2', [self.n_l1, self.n_l2], initializer=tf.contrib.layers.xavier_initializer())
            self.W3 = tf.get_variable('W3', [self.n_l2, self.n_l3], initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable('b1', [self.n_l1], initializer=tf.constant_initializer(0.01))
            self.b2 = tf.get_variable('b2', [self.n_l2], initializer=tf.constant_initializer(0.01))
            self.b3 = tf.get_variable('b3', [self.n_l3], initializer=tf.constant_initializer(0.00))

            self.layer1 = tf.nn.relu(tf.matmul(self.input, self.W1) + self.b1)
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
            self.layer3 = tf.matmul(self.layer2, self.W3) + self.b3

            self.predictions = tf.argmax(self.layer3, axis=1) + 1

            self.labels = tf.placeholder(tf.int32, [None], name='labels')
            self.labels = self.labels - 1
            self.labels_onehot = tf.one_hot(indices=self.label, depth=26)

            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.labels_onehot, self.layer3))

            self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
