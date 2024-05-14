import sys
import time
from sklearn.externals import joblib

# Add the Scripts folder to the system path
sys.path.append('./Scripts')

# Import necessary functions and models
from pr_data_preparation import preprocess_image, compute_sift_descriptors, compute_histograms
from pr_model_training import test_svm
from pr_save_load_models import load_Kmeans_model, load_SVM_model

# Load the trained models
kmeans = load_Kmeans_model()
clf = load_SVM_model()

# Load the trained scaler
scaler = joblib.load('scaler.joblib')

# Import FastAPI related modules
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Create a FastAPI instance
app = FastAPI() 

@app.post("/predict")
async def predict(img : UploadFile = File(...)):
    try : 
        # Record the start time
        start_time = time.time()

        # Preprocess the image
        preprocessed_image = preprocess_image(img)

        # Compute SIFT descriptors
        sift_descriptors_Kmeans_test, sift_descriptors_SVM_test = compute_sift_descriptors([preprocessed_image])

        # Compute histograms
        feature_vectors_test = compute_histograms(kmeans, sift_descriptors_SVM_test)

        # Scale the feature vectors using the trained scaler
        scaled_feature_vectors_test = scaler.transform(feature_vectors_test)

        # Make a prediction
        prediction = clf.predict(scaled_feature_vectors_test)

        # Calculate the time taken
        time_taken = time.time() - start_time

        print("Prediction: ", prediction)
        print("Time taken: ", time_taken)

        # Return the prediction and the time taken
        return JSONResponse( content = {"Prediction": str(prediction[0]) , "Time Taken": time_taken} , status_code=200)
    except Exception as e:
        # Return the error message in case of an exception
        return JSONResponse(content={"error":str(e)},status_code=500)