import pandas as pd
import numpy as np


class Autoencoder:
    def __init__(self, x, y, hidden_units, epochs, lr):
        self.x = x
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.output_size = y.shape[1]
        self.weights_one = np.random.rand(self.x.shape[1], hidden_units)
        self.output_one = np.zeros(hidden_units)
        self.weights_two = np.random.rand(hidden_units, self.output_size)
        self.output_two = np.zeros(self.output_size)

    def feedforward(self):
        self.output_one = sigmoid(np.dot(self.x, self.weights_one))
        self.output_two = sigmoid(np.dot(self.output_one, self.weights_two))

    def backpropagate(self):
        print('Loss: ' + str(np.sum(np.sum(np.square(self.y - self.output_two)))))

        gradient_two = np.dot(self.output_one.T,
                              (2*(self.y - self.output_two) * sigmoid_prime(self.output_two))
                              ) * self.lr

        gradient_one = np.dot(self.x.T,
                              (np.dot(
                                  2*(self.y - self.output_two) * sigmoid_prime(self.output_two),
                                  self.weights_two.T
                              )
                               * sigmoid_prime(self.output_one))
                              ) * self.lr

        self.weights_one += gradient_one
        self.weights_two += gradient_two

    def fit(self):
        for i in range(self.epochs):
            print('Iteration: ' + str(i))
            self.feedforward()
            self.backpropagate()
            print(np.round(model.output_two))
            if (np.round(model.output_two) == y_train.values).all():
                break

    def predict(self):
        self.fit()


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def sigmoid(x):
    return 1/(1+np.exp(-x))


def normalize(col):
    return(col-col.min())/(col.max()-col.min())


data = pd.read_csv('auto_encoder_data.csv')

x_train = data.iloc[:, 0:8]
y_train = data.iloc[:, 8:]
model = Autoencoder(x_train.values, y_train, 3, 5000, .9)
model.fit()




"""
1b
The purpose of the training algorithm is to learn a lower dimension representation of the input data, and then
from that encoded representation get the original input value. The algorithm achieves that by repeated trial and
error. Trying to encode and decode, seeing how much it is off by, then adjust using its error and eventually getting a 
perfect encoding and decoding. 
"""














