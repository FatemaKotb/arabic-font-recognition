from joblib import dump
from joblib import load

def save_Kmeans_model(kmeans):
    # Save the KMeans model
    dump(kmeans, 'kmeans_model.joblib')



def save_SVM_model(svm):
    # Save the SVM model
    dump(svm, 'svm_model.joblib')



def load_Kmeans_model():
    # Load the KMeans model
    return load('kmeans_model.joblib')



def load_SVM_model():
    # Load the SVM model
    return load('svm_model.joblib')