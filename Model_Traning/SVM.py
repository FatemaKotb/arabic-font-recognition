from sklearn import svm
import numpy as np
import joblib

def train_svm(features, labels, gamma='scale', C=6.8, kernel='linear'):
    clf = svm.SVC(gamma=gamma, C=C , kernel=kernel)
    clf.fit(features, labels)
    return clf

def clear_memory(*args):
    for var in args:
        del var

def load_data(filename):
    return np.load(filename)

def save_data(data, filename):
    np.save(filename, data)

def save_model(model, filename):
    joblib.dump(model, filename)