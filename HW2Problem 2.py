import numpy as np
import pandas as pd

percep_data = pd.read_csv('perceptronData.txt', header = None, delim_whitespace = True)

def normalize(col, mean=None, std=None):
    if mean is None and std is None:
        x = (col - np.mean(col)) / np.std(col)
        return x
    else:
        x = (col - mean) / std
        return x

class Perceptron:

    def __init__(self, train_data, lr=.001):
        x = train_data.copy()
        x.loc[x[x.columns[-1]] == -1] *= -1
        self.lr = lr
        self.means = {}
        self.stds = {}
        for col in x[x.columns[:-1]]:
            self.stds[col] = np.std(x[col])
            self.means[col] = np.mean(x[col])
            x[col] = normalize(x[x.columns[:-1]][col])
        self.train_x = x[x.columns[:-1]]
        self.train_y = x[x.columns[-1]].values
        ones = np.ones(shape=(len(self.train_x), 1))
        new_x = np.concatenate((ones, self.train_x), 1)
        self.w = np.random.rand(new_x.shape[1])

    def fit(self):

        x = self.train_x
        y = self.train_y
        ones = np.ones(shape=(len(y), 1))
        x = pd.DataFrame(np.concatenate((ones, x), 1))

        no_errors = False
        i = 0
        while not no_errors:
            i+=1
            print('iteration :'+str(i))
            preds = x.dot(self.w)

            wrong = [i for i,x in enumerate(preds) if x < 0]


            print('weights: ' + str(self.w))
            print('n_wrong: ' +str(len(wrong)) +'\n')
            sum_x = []
            if len(wrong) == 0:
                no_errors = True
            for col in x.columns:
                wrongs = x.loc[wrong]
                sum_x.append(np.sum(wrongs[col].values))
            self.w = self.w + (self.lr * np.array(sum_x))



perceptron = Perceptron(percep_data)
perceptron.fit()

n_w = perceptron.w
last, n_w = n_w[-1], n_w[:-1]
n_w = n_w/last

print('Final Normalized Weights: ' + str(n_w))