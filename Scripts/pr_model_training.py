from sklearn import svm
from sklearn.preprocessing import StandardScaler

from joblib import dump

def prepare_data_for_svm(feature_vectors):
    # It's a good practice to scale your data before using SVM
    scaler = StandardScaler()
    scaled_feature_vectors = scaler.fit_transform(feature_vectors)

    # Save the scaler
    dump(scaler, 'scaler.joblib')

    return scaled_feature_vectors


def train_svm(x_date, y_data):

    clf = svm.SVC(kernel='linear')

    # Train the model using the training sets
    clf.fit(x_date, y_data)

    return clf



def test_svm(clf, x_test, y_test):
    return clf.score(x_test, y_test)