import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.util import random_noise 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# Load Image
imageName = 'images (1).jpg'
image = plt.imread(imageName)
plt.figure(dpi=150)
plt.title("Original Image")
plt.imshow(image)

# Reshape and Scale Data
pixels = image.reshape(-1, 3)  # Flatten the image into (num_pixels, 3)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_pixels = scaler.fit_transform(pixels)

# Apply KMeans Clustering
n_clusters = 5  # Number of clusters for segmentation
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_pixels)
clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]  # Map each pixel to its cluster center

# Rescale and Reshape to Original Image
segmented_image = scaler.inverse_transform(clustered_pixels)  # Inverse scaling
segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)  # Reshape to image shape

# Display Segmented Image
plt.figure(dpi=150)
plt.title(f"Segmented Image with {n_clusters} Clusters")
plt.imshow(segmented_image)
plt.axis('off')  # Hide axis for better visualization
plt.show()
