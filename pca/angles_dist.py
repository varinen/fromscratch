# PACKAGE: DO NOT EDIT
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import scipy

import sklearn
from sklearn.datasets import fetch_mldata
from ipywidgets import interact

MNIST = fetch_mldata('MNIST original', data_home='./MNIST')


# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def distance(x, y):
    """Compute distance between two vectors x, y using the dot product"""
    x = np.array(x, dtype=np.float).ravel()  # ravel() "flattens" the ndarray
    y = np.array(y, dtype=np.float).ravel()

    dif = x - y
    distance = np.sqrt(dif.T @ dif)
    return distance


def angle(x, y):
    """Compute the angle between two vectors x, y using the dot product"""
    angle = np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)  # -> cosine of the angle
    angle = np.arccos(np.clip(angle, -1, 1))
    return angle


# ===YOU SHOULD EDIT THIS FUNCTION===
def pairwise_distance_matrix_1(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)

    Returns
    --------
    D: matrix of shape (N, M), each entry D[i,j] is the distance between
    X[i,:] and Y[j,:] using the dot product.
    """
    N, D = X.shape
    M, _ = Y.shape

    x_sq_sum = np.sum(X ** 2, axis=1).astype(float)
    y_sq_sum = np.sum(Y ** 2, axis=1).astype(float)
    x_times_y = np.sum((X * Y), axis=1).astype(float)

    x_sq_plus_y_sq = x_sq_sum + y_sq_sum[:, None]

    distance_matrix_1 = np.abs(x_sq_plus_y_sq.T - 2 * x_times_y)
    distance_matrix = np.sqrt(distance_matrix_1)

    return distance_matrix


def pairwise_distance_matrix(X, Y):
    distance_matrix = sklearn.metrics.pairwise.euclidean_distances(X, Y)
    return distance_matrix


d = pairwise_distance_matrix(MNIST.data[:1], MNIST.data)
d1 = d[:, 1:]

means = {}
for n in np.unique(MNIST.target).astype(np.int):
    means[n] = np.mean(MNIST.data[MNIST.target==n], axis=0)

MD = np.zeros((10, 10))
AG = np.zeros((10, 10))
for i in means.keys():
    for j in means.keys():
        MD[i, j] = distance(means[i], means[j])
        AG[i, j] = angle(means[i], means[j])
a = 1
