import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def perform_grid_search(features_pca, y_data):
    param_dist = {
    'C': np.arange(6, 8.1, 0.1),       
    'gamma': ['scale','auto'],       
    'kernel': ['linear', 'rbf', 'poly'],
    }
    clf = SVC()
    grid_search = RandomizedSearchCV(clf, param_dist, n_iter=10, cv=5, n_jobs=-1)
    grid_search.fit(features_pca, y_data)
    return grid_search

def print_best_params(grid_search):
    print("Search Done - Best Parameters:", grid_search.best_params_, "Score:", grid_search.best_score_)
