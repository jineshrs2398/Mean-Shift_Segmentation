import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import warnings

warnings.filterwarnings("ignore")

# Load Image
# imageName = 'images (1).jpg'
imageName = 'wallpaper_image.jpg'

image = plt.imread(imageName)


# Reshape and Scale Data
pixels = image.reshape(-1, 3)  # Flatten the image into (num_pixels, 3)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_pixels = scaler.fit_transform(pixels)

# Apply KMeans Clustering
n_clusters = 5  # Number of clusters for segmentation
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_pixels)
clustered_pixels_kmeans = kmeans.cluster_centers_[kmeans.labels_]  # Map each pixel to its cluster center

# Rescale and Reshape to Original Image for KMeans
segmented_image_kmeans = scaler.inverse_transform(clustered_pixels_kmeans)  # Inverse scaling
segmented_image_kmeans = segmented_image_kmeans.reshape(image.shape).astype(np.uint8)  # Reshape to image shape

# Apply Mean Shift Segmentation
bandwidth = estimate_bandwidth(scaled_pixels, quantile=0.2, n_samples=500)
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(scaled_pixels)
clustered_pixels_mean_shift = mean_shift.cluster_centers_[mean_shift.labels_]  # Map each pixel to its cluster center

# Rescale and Reshape to Original Image for Mean Shift
segmented_image_mean_shift = scaler.inverse_transform(clustered_pixels_mean_shift)  # Inverse scaling
segmented_image_mean_shift = segmented_image_mean_shift.reshape(image.shape).astype(np.uint8)  # Reshape to image shape

kmeans_gray = cv2.cvtColor(segmented_image_kmeans, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
_, kmeans_binary = cv2.threshold(kmeans_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours_kmeans, _ = cv2.findContours(kmeans_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented_with_lines_kmeans = segmented_image_kmeans.copy()
cv2.drawContours(segmented_with_lines_kmeans, contours_kmeans, -1, (255, 255, 255), 3)  # Draw blue lines

mean_shift_gray = cv2.cvtColor(segmented_image_mean_shift, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
_, mean_shift_binary = cv2.threshold(mean_shift_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours_mean_shift, _ = cv2.findContours(mean_shift_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented_with_lines_mean_shift = segmented_image_mean_shift.copy()
cv2.drawContours(segmented_with_lines_mean_shift, contours_mean_shift, -1, (255, 255, 255), 3)  # Draw blue lines


# Plot Results
plt.figure()

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

# KMeans Segmentation with Lines
plt.subplot(1, 3, 2)
plt.title("KMeans Segmentation")
plt.imshow(segmented_with_lines_kmeans)
plt.axis('off')

# Mean Shift Segmentation with Lines
plt.subplot(1, 3, 3)
plt.title("Mean Shift Segmentation")
plt.imshow(segmented_with_lines_mean_shift)
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentation.jpg')
plt.show()

