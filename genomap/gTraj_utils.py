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
def nearest_divisible_by_four(num):
    remainder = num % 4

    if remainder == 0:
        #print(f"The number {num} is divisible by 4.")
        return num
    else:
        #print(f"The number {num} is not divisible by 4. Setting to nearest number divisible by 4.")
        return num + (4 - remainder)