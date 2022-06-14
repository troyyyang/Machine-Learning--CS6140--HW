import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np


def normalize(col):
    return(col-col.min())/(col.max()-col.min())


def accuracy(actual, pred):
    true = sum(actual[i] == round(pred[i] + .1) for i in range(len(pred)))
    return true/len(pred)


wine_train = pd.read_csv('train_wine.csv', header = None)
wine_test = pd.read_csv('test_wine.csv', header = None)

x_train = wine_train.iloc[:, 1:]
x_train = normalize(x_train)

x_test = wine_test.iloc[:, 1:]
x_test = normalize(x_test)

y_label_train = wine_train.iloc[:, 0]
y_label_test = wine_test.iloc[:, 0]

y_train = pd.get_dummies(y_label_train)

# Keras
model = tf.keras.Sequential()
model.add(keras.layers.Dense(5, activation='sigmoid', input_shape=(13,)))

model.add(keras.layers.Dense(3, activation='sigmoid'))


optimizer = tf.train.GradientDescentOptimizer(0.8)

model.compile(loss='mean_squared_error', optimizer=optimizer,  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1500, batch_size=100)
preds = (np.argmax(model.predict(x_test), axis=1) + 1).tolist()
print(f'Test Accuracy: {str(accuracy(y_label_test, preds))}')


# tensorflow


hidden_nodes = 100
epochs = 2000

X = tf.placeholder(shape=(151, 13), dtype=tf.float32, name='X')
y = tf.placeholder(shape=(151, 3), dtype=tf.float32, name='y')

W1 = tf.Variable(tf.truncated_normal(shape=[13, hidden_nodes], dtype=tf.float32))
W2 = tf.Variable(tf.truncated_normal(shape=[hidden_nodes, 3], dtype=tf.float32))


l1 = tf.nn.sigmoid(tf.matmul(X, W1))
output = tf.nn.sigmoid(tf.matmul(l1, W2))

loss = tf.reduce_mean(tf.squared_difference(output, y))
train = tf.train.GradientDescentOptimizer(.9).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(epochs):
    _, error, out = sess.run(fetches=[train, loss, output], feed_dict={X: x_train, y: y_train})
    print(error)


def sigmoid(x):
    return 1/(1+np.exp(-x))


weights1 = sess.run(W1)
weights2 = sess.run(W2)

raw_pred = sigmoid(np.dot(sigmoid(np.dot(x_test, weights1)), weights2))
preds = (np.argmax(raw_pred, axis=1) + 1).tolist()


print(f'Test Accuracy: {str(accuracy(y_label_test, preds))}')






