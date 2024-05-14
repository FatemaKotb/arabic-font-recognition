from PIL import Image
import io
import numpy as np

import sys
import time
import joblib

# Add the Scripts folder to the system path
sys.path.append('./Scripts')

# Import necessary functions and models
from pr_data_preparation import preprocess_image
from pr_model_training import test_svm
from pr_save_load_models import load_Kmeans_model, load_SVM_model
from pr_feature_extraction import compute_sift_descriptors , compute_histograms

# Load the trained models
kmeans = load_Kmeans_model()
clf = load_SVM_model()

# Load the trained scaler
# scaler = joblib.load('scaler.joblib')

# Import FastAPI related modules
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "SHOGHL FAKHER MEN EL AKHER!"}

@app.post("/test")
async def test(file : UploadFile = File(...)):
    try:
        start_time = time.time()
        img=await file.read()
        test_image = Image.open(io.BytesIO(img))
        img = np.array(test_image)

        preprocessed_image = preprocess_image(img).astype('uint8')
        sift_descriptors_Kmeans_test, sift_descriptors_SVM_test = compute_sift_descriptors([preprocessed_image])
        feature_vectors_test = compute_histograms(kmeans, sift_descriptors_SVM_test)
        # scaled_feature_vectors_test = scaler.transform(feature_vectors_test)
        prediction = clf.predict(feature_vectors_test)
        # Calculate the time taken
        time_taken = time.time() - start_time
        return JSONResponse(content={"result":str(prediction[0]), "time_taken": time_taken},status_code=200)
    except Exception as e:
        # Return the error message in case of an exception
        return JSONResponse(content={"error":str(e)},status_code=500)