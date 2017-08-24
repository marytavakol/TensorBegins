
import numpy as np
import tensorflow as tf
import scipy.io



mat = scipy.io.loadmat('PenBasedRec_16f.mat')
tr = mat['Xtrain']
te = mat['Xtest']

[nf, nTr, nc] = np.shape(tr)
nTe = np.shape(te)[1]

#Normalization
train_x = np.empty((0, nf))
test_x = np.empty((0, nf))
train_y = np.empty((0, nc))
test_y = np.empty((0, nc))
for i in range(nc):
    minvec = np.min(tr[:, :, i], axis=1)
    maxvec = np.max(tr[:, :, i], axis=1)
    train_x = np.vstack((train_x, (np.divide(tr[:, :, i] - np.reshape(minvec, (nf, 1)), np.reshape(maxvec, (nf, 1))-np.reshape(minvec, (nf, 1)))).T))
    minvec = np.min(te[:, :, i], axis=1)
    maxvec = np.max(te[:, :, i], axis=1)
    test_x = np.vstack((test_x, np.divide(te[:, :, i] - np.reshape(minvec, (nf, 1)), np.reshape(maxvec, (nf, 1))-np.reshape(minvec, (nf, 1))).T))
    t = np.zeros((1, nc)) - 0.9
    t[:, i] = 0.9
    train_y = np.vstack((train_y, np.tile(t, (nTr, 1))))
    test_y = np.vstack((test_y, np.tile(t, (nTe, 1))))

nn = 15
lr = 0.01

w_1 = tf.Variable(tf.random_uniform([nf, nn], -0.5, 0.5))
w_2 = tf.Variable(tf.random_uniform([nn, nc], -0.5, 0.5))
b_1 = tf.Variable(tf.random_uniform([nn], -0.5, 0.5))
b_2 = tf.Variable(tf.random_uniform([nc], -0.5, 0.5))

X = tf.placeholder("float", shape=[None, nf])
y = tf.placeholder("float", shape=[None, nc])

hid = tf.nn.tanh(tf.add(tf.matmul(X, w_1), b_1))
yhat = tf.nn.tanh(tf.add(tf.matmul(hid, w_2), b_2))
predict = tf.argmax(yhat, axis=1)
cost = tf.reduce_mean(tf.squared_difference(y, yhat))
#cost = tf.reduce_mean(tf.nn.l2_loss(y - yhat))
#cost = tf.nn.l2_loss(y - yhat)
updates = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(100):
    #shuffle
    s = np.arange(nTr*nc)
    np.random.shuffle(s)
    train_x_sh = train_x[s, :]
    train_y_sh = train_y[s, :]
    for i in range(len(train_x_sh)):
        sess.run(updates, feed_dict={X: train_x_sh[i: i + 1], y: train_y_sh[i: i + 1]})

    train_accuracy = np.mean(np.argmax(train_y_sh, axis=1) == sess.run(predict, feed_dict={X: train_x_sh, y: train_y_sh}))
    test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_x, y: test_y}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
sess.close()

print('done')