from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump, load
import numpy as np

def load_model(model_path):
    return load(model_path)

def load_data(data_path):
    return np.load(data_path)

def transform_data(pca, data):
    return pca.transform(data)

def make_predictions(model, data):
    return model.predict(data)

def display_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Confusion Matrix:\n {conf_matrix}")
    print(f"Classification Report:\n {class_report}")
    



# Usage:
# main()