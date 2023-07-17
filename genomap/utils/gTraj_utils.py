"""
Created on Sun Jul 16 21:27:17 2023
@author: Md Tauhidul Islam, Research Scientist, Dept. of radiation Oncology, Stanford University
"""

import numpy as np
from scipy.spatial.distance import cdist

def compute_cluster_distances(X, labels):
    """
    Compute correlation distances from each data point to all cluster centers.

    Parameters:
    X (numpy.ndarray): data matrix where each row is a data point.
    labels (numpy.ndarray): vector of cluster labels for the data points.

    Returns:
    distances (numpy.ndarray): distance matrix where each row corresponds to a 
    data point and each column corresponds to a cluster. The value at (i, j) is 
    the correlation distance from the i-th data point to the j-th cluster center.
    """
    
    # Get unique labels
    unique_labels = np.unique(labels)

    # Compute mean of each class
    means = np.array([X[labels == i].mean(axis=0) for i in unique_labels])

    # Initialize a matrix to store the distances
    distances = np.zeros((X.shape[0], len(unique_labels)))

    # Compute the correlation distance of each data point to all cluster centers
    for i in unique_labels:
        # Compute correlation distances and store them in the corresponding column of the distances matrix
        distances[:, i] = cdist(X, means[i].reshape(1, -1), metric='correlation').ravel()

    return distances

# Usage:
# Assuming X is your matrix and labels is your cluster label vector
# X = np.array([...])  
# labels = np.array([...])

# distances = compute_cluster_distances(X, labels)
# print(distances)
def rgb2gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
def gray2rgb(image):
    return np.repeat(image, 3, axis=3)
def create_sorted_vectors(images):
    num_images = len(images)

    # Create empty lists to hold the values and positions
    values = []
    positions = []

    # Iterate over the images
    for i in range(num_images):
        # Flatten the image to a 1D array and get the sorted indices
        flat_image = images[i].flatten()
        sorted_indices = np.argsort(flat_image)[::-1]  # sort in descending order

        # Save the sorted values and positions
        values.append(flat_image[sorted_indices])
        positions.append(np.unravel_index(sorted_indices, images[i].shape))

    # Convert the lists to numpy arrays
    values = np.array(values)
    positions = np.array(positions)
    return values, positions
    
def sort_image_by_positions(image, positions):
    # Flatten the image to a 1D array
    flat_image = image.flatten()
    # Reorder the flattened image using the positions
    sorted_image = flat_image[np.ravel_multi_index(positions, image.shape)]  

    return sorted_image
def nearest_divisible_by_four(num):
    remainder = num % 4

    if remainder == 0:
        #print(f"The number {num} is divisible by 4.")
        return num
    else:
        #print(f"The number {num} is not divisible by 4. Setting to nearest number divisible by 4.")
        return num + (4 - remainder)