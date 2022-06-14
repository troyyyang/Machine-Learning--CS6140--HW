import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np


def normalize(col):
    return(col-col.min())/(col.max()-col.min())


data = pd.read_csv('auto_encoder_data.csv')

x_train = data.iloc[:, 0:8]
y_train = data.iloc[:, 8:]


# keras
model = tf.keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(8,)))

model.add(keras.layers.Dense(8, activation='sigmoid'))


optimizer = tf.train.GradientDescentOptimizer(0.8)

model.compile(loss='mean_squared_error', optimizer=optimizer,  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=100)


# tensorflow
hidden_nodes = 4
epochs = 19000

X = tf.placeholder(shape=(8, 8), dtype=tf.float32, name='X')
y = tf.placeholder(shape=(8, 8), dtype=tf.float32, name='y')

W1 = tf.Variable(tf.truncated_normal(shape=[8, hidden_nodes], dtype=tf.float32))
W2 = tf.Variable(tf.truncated_normal(shape=[hidden_nodes, 8], dtype=tf.float32))

l1 = tf.nn.sigmoid(tf.matmul(X, W1))
output = tf.nn.sigmoid(tf.matmul(l1, W2))

loss = tf.reduce_mean(tf.squared_difference(output, y))
train = tf.train.GradientDescentOptimizer(.99).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(epochs):
    _, error, out = sess.run(fetches=[train, loss, output], feed_dict={X: x_train, y: y_train})
    print(error)
    print(np.round(out))










