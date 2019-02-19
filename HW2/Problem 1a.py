
import numpy as np
import pandas as pd

house_train = pd.read_csv('housing_train.txt', header = None, delim_whitespace = True)
house_test = pd.read_csv('housing_test.txt', header = None, delim_whitespace = True)

spambase_data_txt = pd.read_csv('spambase.data.txt', header = None)


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

def accuracy(actual, pred):
    true = 0
    for i in range(len(pred)):
        if actual[i] == round(pred[i] + .1):
            true +=1
    return true/len(pred)

class RidgeRegression:


    def __init__(self, train_data, l=.5):
        x = train_data.copy()
        self.l = l
        self.means = {}
        self.stds = {}  
        for col in x[x.columns[:-1]]:
            self.stds[col] = np.std(x[col])
            self.means[col] = np.mean(x[col])
            x[col] = normalize(x[x.columns[:-1]][col])
        self.train_x  = x[x.columns[:-1]]
        self.train_y = x[x.columns[-1]].values
        self.w = None
        self.w0 = None


    def fit(self):
        x = self.train_x
        y = self.train_y
        inside_paran = x.T.dot(x) + self.l*np.eye(x.shape[1])
        self.w = np.linalg.inv(inside_paran).dot(x.T.dot(y))
        self.w0 = np.mean(y)
        

    def predict(self, x):
        return x.dot(self.w) + self.w0



def normalize(col, mean = None, std = None):
    
    if mean is None and std is None:
        x = (col- np.mean(col))/np.std(col)
        return x
    else:
        x = (col - mean)/std
        return x

def normalize_test_data(test_x, model):
    x = test_x.copy()
    for col in test_x:
        x[col] = normalize(test_x[col], model.means[col], model.stds[col])
    return x


ridge = RidgeRegression(house_train, l = .9)
ridge.fit()
house_test_x =house_test[house_test.columns[:-1]]
house_test_x = normalize_test_data(house_test_x, ridge)
house_test_y = list(house_test[house_test.columns[-1]].values)
preds = ridge.predict(house_test_x)
preds_train = ridge.predict(ridge.train_x)

print('housing test error: '+ str(mse(house_test_y, preds)))
print('housing train error: '+ str(mse(ridge.train_y, preds_train)))



folds = k_fold(spambase_data_txt, 3)
test_accs = []
train_accs = []
test_mses = []
for fold in folds:


    t = pd.concat([fold, spambase_data_txt])

    t = t.drop_duplicates(keep=False)

    ridge = RidgeRegression(t, l=0)

    spam_test_x = normalize_test_data(fold[fold.columns[:-1]], ridge)
    spam_test_y = fold[fold.columns[-1]]
    ridge.fit()
    preds = ridge.predict(spam_test_x)

    train_acc = accuracy(list(ridge.train_y), list(ridge.predict(ridge.train_x)))
    train_accs.append(train_acc)
    test_acc = accuracy(list(spam_test_y), list(preds))
    test_accs.append(test_acc)
    test_mses.append(mse(list(spam_test_y), list(preds)))

print('spambase data')
print('average test accuracy: ' + str(np.mean(test_accs)))
print('average train accuracy: ' +str(np.mean(train_accs)))












