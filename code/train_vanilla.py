import numpy as np
import tensorflow as tf
from nets import vanilla_network
import scipy.io as sio
from read_data import load_data

def train_batch(net, sess, X, Y):
    _, loss = sess.run([net.optimize, net.loss], feed_dict={net.raw_input: X, net.labels: Y})
    return loss

def predict(net, sess, X):
    Y_predict = sess.run(net.predictions, feed_dict={net.raw_input: X})
    return Y_predict

'''
def show(i):
    if i < 10:
        return '00' + str(i)
    if i >= 10 and i < 100:
        return '0' + str(i)
    return str(i)
'''

if __name__ == '__main__':
    # read file and get X_train, Y_train, X_val, Y_val, n_train, n_val
    '''
    data = sio.loadmat("emnist-letters.mat")
    X_train = data['dataset'][0][0][0][0][0][0]
    s = ''
    cnt = 0
    for i in X_train[0]:
    	s = s + show(i) + ' '
    	cnt += 1
    	if cnt % 28 == 0:
    		print(s)
    		s = ''
    X_train = X_train * 1.0 / 255
    Y_train = data['dataset'][0][0][0][0][0][1]
    print(Y_train[0])
    n_train = 124800
    X_val = data['dataset'][0][0][1][0][0][0]
    X_val = X_val * 1.0 / 255
    Y_val = data['dataset'][0][0][1][0][0][1]
    n_val = 20800
    '''
    X, Y, _, _ = load_data('emnist-letters.mat')
    X_train, Y_train = X
    '''
    s = ''
    cnt = 0
    for i in range(28):
        for j in range(28):
            s = s + show(int(X_train[0][i][j]*255)) + ' '
            cnt= cnt + 1
            if cnt % 28 == 0:
                print(s)
                s = ''
    print(Y_train[0])
    '''
    n_train = X_train.shape[0]
    X_val, Y_val = Y
    n_val = X_val.shape[0]
    # print(X_train[0])
    # print(Y_train[0])
    #print(n_train, n_val)

    net = vanilla_network('vanilla_network', lr=0.01)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('loss_curve', sess.graph)

    batch_size = 64

    for epoch in range(100000):
        indices = np.random.choice(n_train, batch_size)
        X_batch = X_train[indices]
        Y_batch = Y_train[indices]
        loss = train_batch(net, sess, X_batch, Y_batch)

        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
        file_writer.add_summary(summary, epoch)

        if epoch % 1000 == 0:
            Y_predict = predict(net, sess, X_val)
            test_acc = np.mean(Y_predict == Y_val)
            Y_predict = predict(net, sess, X_train)
            train_acc = np.mean(Y_predict == Y_train)
            print("epoch = %d, train_acc = %f, test_acc = %f, loss = %f" %(epoch, train_acc, test_acc, loss))

        '''
        if epoch % 5000 == 0:
            Y_predict = predict(net, sess, X_train)
            acc = np.mean(Y_predict == Y_train)
            print("************epoch = %d, acc = %f" %(epoch, acc))
        '''
