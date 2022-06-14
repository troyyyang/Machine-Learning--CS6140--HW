import pandas as pd
import numpy as np


class NeuralNet:
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
        print(f'Loss: {str(np.sum(np.sum(np.square(self.y - self.output_two))))}')

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
            print(f'Iteration: {str(i)}')
            self.feedforward()
            self.backpropagate()
            preds = (np.argmax(model.output_two, axis=1) + 1).tolist()
            print(f'Accuracy: {str(accuracy(y_label_train, preds))}')
            if accuracy(y_label_train, preds) == 1:
                break


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def sigmoid(x):
    return 1/(1+np.exp(-x))


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
model = NeuralNet(x_train.values, y_train, 600, 1000, .3)
model.fit()
model.x = x_test
model.feedforward()
preds = (np.argmax(model.output_two, axis=1) + 1).tolist()
print(f'Test Accuracy: {str(accuracy(y_label_test, preds))}')












