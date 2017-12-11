import numpy as np
import tensorflow as tf
from nets import skip_network

def train_batch(net, sess, X, Y):
	sess.run(net.optimize, feed_dict={net.input: X, net.labels: Y})

def predict(net, sess, X):
	Y_predict = sess.run(net.predictions, feed_dict={net.input: X})
	return Y_predict

if __name__ == '__main__':
	# read file and get X_train, Y_train, X_val, Y_val, n_train, n_val
	X_train = data[0][0][0][0][0]
	Y_train = data[0][0][0][0][1]
	n_train = 124800
	X_val = data[0][0][1][0][0]
	Y_val = data[0][0][1][0][1]
	n_val = 20800

	net = skip_network('skip_network', lr=0.001)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

    batch_size = 256

	for epoch in range(100000):
		indices = np.random.choice(n_train, batch_size)
		X_batch = X_train[indices]
		Y_batch = Y_train[indices]
		train_batch(net, sess, X_batch, Y_batch)

		if epoch % 100 == 0:
			Y_predict = predict(net, sess, X_val)
			acc = np.reduce_mean(Y_predict == Y_val)
			print("epoch = %d, acc = %f" %(epoch, acc))
