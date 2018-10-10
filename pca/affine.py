# PACKAGE: DO NOT EDIT
import numpy as np


def mean(X):
    """Compute the mean for a dataset

    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.

    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    # mean = np.sum(X, axis=0) / X.shape[0]
    mean = np.mean(X, axis=0)
    return mean


def cov(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.

    Returns
    -------
    covariance_matrix: (D, D) ndarray which is the covariance matrix of the dataset.

    """
    # It is possible to vectorize our code for computing the covariance, i.e. we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    N, D = X.shape

    covariance_matrix = np.cov(X, rowvar=False, bias=True)
    return covariance_matrix


def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        mean: ndarray, the mean vector
        A, b: affine transformation applied to x. i.e. Ax + b
    Returns:
        mean vector after affine transformation
    """
    affine_m = A @ mean + b
    return affine_m


def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X
    Returns:
        covariance matrix after the transformation
    """
    affine_cov = A @ S @ A.T  # EDIT THIS
    return affine_cov


random = np.random.RandomState(42)
A = random.randn(4,4)
b = random.randn(4)

X = random.randn(100, 4)

X1 = ((A @ (X.T)).T + b)  # applying affine transformation once
X2 = ((A @ (X1.T)).T + b) # and again

np.testing.assert_almost_equal(mean(X1), affine_mean(mean(X), A, b))
np.testing.assert_almost_equal(cov(X1),  affine_covariance(cov(X), A, b))
print('correct')