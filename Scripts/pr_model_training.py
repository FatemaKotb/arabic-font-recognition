from sklearn import svm
from sklearn.preprocessing import StandardScaler

def prepare_data_for_svm(feature_vectors):
    # It's a good practice to scale your data before using SVM
    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)

    return feature_vectors



def train_svm(x_date, y_data):
    
#     # Linear Kernel
#     clf_linear = svm.SVC(kernel='linear')

#     # Polynomial Kernel
#     clf_poly = svm.SVC(kernel='poly', degree=3)  # degree is a parameter for the polynomial kernel, you can adjust it as needed

#     # Radial Basis Function (RBF) Kernel
#     clf_rbf = svm.SVC(kernel='rbf', gamma='scale')  # gamma is a parameter for the RBF kernel, 'scale' is usually a good default value

#     # Sigmoid Kernel
#     clf_sigmoid = svm.SVC(kernel='sigmoid')

    clf = svm.SVC(kernel='linear')

    # Train the model using the training sets
    clf.fit(x_date, y_data)

    return clf



def test_svm(clf, x_test, y_test):
    return clf.score(x_test, y_test)