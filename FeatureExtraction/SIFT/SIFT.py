import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

def compute_sift_descriptors_to_train_Kmeans(x_data):
    # Initialize the SIFT feature extractor
    sift = cv2.SIFT_create()

    # Initialize an empty list to store the SIFT descriptors
    sift_descriptors = []

    # Iterate over each image in x_data_subset
    for img in tqdm(x_data, desc="Computing SIFT descriptors"):
        # Compute the SIFT descriptors for the current image
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Check if descriptors is None
        if descriptors is not None:
            # If descriptors is not None, append the descriptors
            sift_descriptors.extend(descriptors)

    return sift_descriptors


# Choose between Kmeans and MiniBatchKMeans
def train_kmeans(sift_descriptors, model_type="MiniBatchKMeans"):
    
    if model_type == "MiniBatchKMeans":
        # Initialize the MiniBatchKMeans algorithm
        kmeans = MiniBatchKMeans(n_clusters=100, random_state=42, batch_size=1000)
    else:
        # Initialize the K(Means algorithm
        kmeans = KMeans(n_clusters=100, random_state=42)

    # Fit the KMeans algorithm on the stacked descriptors
    kmeans.fit(sift_descriptors)

    return kmeans



def compute_sift_descriptors_to_train_SVM(x_data):
    # Initialize the SIFT feature extractor
    sift = cv2.SIFT_create()

    # Initialize an empty list to store the SIFT descriptors
    sift_descriptors = []

    # Iterate over each image in x_data_subset
    for img in tqdm(x_data, desc="Computing SIFT descriptors"):
        # Compute the SIFT descriptors for the current image
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Check if descriptors is None
        if descriptors is not None:
            # If descriptors is not None, append the descriptors
            sift_descriptors.append(descriptors)
        else:
            # If descriptors is None, append an empty list
            sift_descriptors.append([])

    return sift_descriptors



def compute_histograms(kmeans, image_descriptors):
    # Initialize an empty list to store the feature vectors
    feature_vectors = []

    # Iterate over each image in image_descriptors
    for descriptors in tqdm(image_descriptors, desc="Computing histograms"):
        # Check if descriptors is not empty
        if len(descriptors) > 0:
            # Predict the cluster assignments for the descriptors of the current image
            cluster_assignments = kmeans.predict(descriptors.astype(np.float64))

            # Create a histogram of cluster assignments
            histogram = np.bincount(cluster_assignments, minlength=100)
        else:
            # If descriptors is empty, create an empty histogram
            histogram = np.zeros(100)

        # Append the histogram to feature_vectors
        feature_vectors.append(histogram)

    return feature_vectors