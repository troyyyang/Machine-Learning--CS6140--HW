
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


house_train = pd.read_csv('housing_train.txt', header = None, delim_whitespace = True)
house_test = pd.read_csv('housing_test.txt', header = None, delim_whitespace = True)
spambase_data_txt = pd.read_csv('spambase.data.txt', header = None)


def confusion_matrix_vals(actual, pred):
    fp,tp,fn,tn = 0,0,0,0
    p = [pr + .1 for pr in pred]
    for i in range(len(p)):
        if actual[i] == round(p[i]) == 0:
            tn +=1
        elif actual[i] == round(p[i]) == 1:
            tp +=1
        elif (actual[i] == 1) and (round(p[i]) == 0):
            fn +=1
        elif (actual[i] == 0) and (round(p[i]) == 1):
            fp +=1
    return fp,tp,fn,tn

def accuracy(actual, pred):
    true = 0
    for i in range(len(pred)):
        if actual[i] == round(pred[i] +.1):
            true +=1
    return true/len(pred)
def k_fold(data, num_folds):

    data_random = data.sample(frac=1)
    folds = []

    fold_size = data.shape[0] // num_folds

    for fold in range(num_folds):
        index_offset = fold * fold_size
        df = pd.DataFrame(data_random.iloc[index_offset:index_offset + fold_size])
        folds.append(df)

    return folds



def mse(actual, pred):
    error = 0.0
    for i in range(len(actual)):
        e = pred[i] - actual[i]
        error += (e ** 2)
    mean_error = error / float(len(actual))
    return np.mean(mean_error)

class LinearRegressionGD:

    def __init__(self, train_data, lr=.5, epochs = 1000):
        x = train_data.copy()
        self.lr =lr
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
        self.w = np.zeros(new_x.shape[1])
        self.epochs = epochs

    def fit(self):

        x = self.train_x
        y = self.train_y
        ones = np.ones(shape=(len(y), 1))
        x = pd.DataFrame(np.concatenate((ones, x), 1))

        for i in range(self.epochs):
            preds = x.dot(self.w)
            errors = preds - y
            thetas = []
            #print(np.mean(np.square(errors)))
            for col in x.columns:
                thetas.append(errors.dot(x[col].values))
            #print(thetas)
            self.w = self.w - (self.lr * np.array(thetas))



    def predict(self, x):
        return x.dot(self.w[1:len(self.w)]) + self.w[0]


def normalize(col, mean=None, std=None):
    if mean is None and std is None:
        x = (col - np.mean(col)) / np.std(col)
        return x
    else:
        x = (col - mean) / std
        return x

def normalize_test_data(test_x, model):
    x = test_x.copy()
    for col in test_x:
        x[col] = normalize(test_x[col], model.means[col], model.stds[col])
    return x

print('housing data')
lgd = LinearRegressionGD(house_train, lr = .0001, epochs = 1000)
lgd.fit()
house_test_x =house_test[house_test.columns[:-1]]
house_test_x = normalize_test_data(house_test_x, lgd)
house_test_y = list(house_test[house_test.columns[-1]].values)
preds = lgd.predict(house_test_x)
preds_train = lgd.predict(lgd.train_x)
print('test error: '+ str(mse(house_test_y, preds)))
print('train error: '+ str(mse(lgd.train_y, preds_train)))


print('spambase data')

folds = k_fold(spambase_data_txt, 3)
test_accs = []
train_accs = []
test_mses = []
fps = []
tps = []
fns = []
tns = []
total = []
pos = []
neg = []

for idx, fold in enumerate(folds):

    print('fold: ' +str(idx+1))

    t = pd.concat([fold, spambase_data_txt])

    t = t.drop_duplicates(keep=False)

    lrgd = LinearRegressionGD(t, lr = .000001, epochs = 3000)

    spam_test_x = normalize_test_data(fold[fold.columns[:-1]], lrgd)
    spam_test_y = fold[fold.columns[-1]]
    lrgd.fit()
    preds = lrgd.predict(spam_test_x)

    train_acc = accuracy(list(lrgd.train_y), list(lrgd.predict(lrgd.train_x)))
    train_accs.append(train_acc)
    test_acc = accuracy(list(spam_test_y), list(preds))

    test_accs.append(test_acc)
    test_mses.append(mse(list(spam_test_y), list(preds)))


    fp, tp, fn, tn = confusion_matrix_vals(list(spam_test_y), list(preds))
    fps.append(fp)
    tps.append(tp)
    fns.append(fn)
    tns.append(tn)



print('average number of fps: ' + str(np.mean(fps)))
print('average number of tps: ' + str(np.mean(tps)))
print('average number of fns: ' + str(np.mean(fns)))
print('average number of tns: ' + str(np.mean(tns)))
print('fpr: ' + str(np.mean(fps) / (np.mean(fps) + (np.mean(tns))) ))
print('tpr: ' + str(np.mean(tps) / (np.mean(tps) + (np.mean(fns))) ))
print('average test accuracy: ' + str(np.mean(test_accs)))
print('average train accuracy: ' +str(np.mean(train_accs)))
print('average test mse: ' + str(np.mean(test_mses)))




