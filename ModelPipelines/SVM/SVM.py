from sklearn import svm
from sklearn.preprocessing import StandardScaler

def prepare_data_for_svm(feature_vectors):
    # It's a good practice to scale your data before using SVM
    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)

    return feature_vectors

def train_svm(x_date, y_data):
    # Create a SVM classifier with a Radial Basis Function (RBF) kernel
    clf = svm.SVC(kernel='rbf', gamma='scale')

    # Train the model using the training sets
    clf.fit(x_date, y_data)

    return clf

def test_svm(clf, x_test, y_test):
    # Predict the response for test dataset
    y_pred = clf.predict(x_test)

    return clf.score(x_test, y_test)
