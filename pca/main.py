# PACKAGE: DO NOT EDIT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people, fetch_mldata, fetch_olivetti_faces
import time
import timeit

# matplotlib.use('Agg')


def mean_naive(X):
    """Compute the mean for a dataset by iterating over the dataset

        Arguments
        ---------
        X: (N, D) ndarray representing the dataset.
            N is the number of samples in the dataset
            and D is the feature dimension of the dataset

        Returns
        -------
        mean: (D, ) ndarray which is the mean of the dataset.
        """
    N, D = X.shape
    mean = np.zeros(D)
    for n in range(N):
        for k in range(D):
            mean[k] += X[n, k]
    mean = mean / N
    return mean


def cov_naive(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
        N is the number of samples in the dataset
        and D is the feature dimension of the dataset

    Returns
    -------
    covariance: (D, D) ndarray which is the covariance matrix of the dataset.

    """
    N, D = X.shape
    covariance = np.zeros((D, D))
    mean = mean_naive(X)
    for n in range(N):
        for k in range(D):
            for l in range(D):
                covariance[k, l] += (X[n, k] - mean[k]) * (X[n, l] - mean[l])

    covariance = covariance / N
    return covariance


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


def test():
    a = np.random.randn(4, 3)

    print(a)

    b = mean_naive(a)

    print(b)

    c = cov_naive(a)

    print(c)

    e = mean(a)

    print(e)

    f = cov(a)

    print(f)


def display_faces(faces, n=0):
    plt.figure()
    plt.imshow(faces[n].reshape((64, 64)), cmap='gray')
    plt.show()


def mean_face(faces):
    """Compute the mean of the `faces`

    Arguments
    ---------
    faces: (N, 64 * 64) ndarray representing the faces dataset.

    Returns
    -------
    mean_face: (64, 64) ndarray which is the mean of the faces.
    """
    mean_face = mean(faces)
    return mean_face


def time(f, repeat=10):
    """A helper function to time the execution of a function.

    Arguments
    ---------
    f: a function which we want to time it.
    repeat: the number of times we want to execute `f`

    Returns
    -------
    the mean and standard deviation of the execution.
    """
    times = []
    for _ in range(repeat):
        start = timeit.default_timer()
        f()
        stop = timeit.default_timer()
        times.append(stop - start)
    return np.mean(times), np.std(times)


def do_faces():
    plt.style.use('fivethirtyeight')

    image_shape = (64, 64)
    # Load faces data
    dataset = fetch_olivetti_faces()

    faces = dataset.data
    print('Shape of the faces dataset: {}'.format(faces.shape))
    print('{} data points'.format(faces.shape[0]))

    display_faces(faces, 10)
    plt.figure()
    plt.imshow(mean_face(faces).reshape((64, 64)), cmap='gray')
    plt.show()

def do_time():
    fast_time = []
    slow_time = []

    for size in np.arange(100, 5000, step=100):
        X = np.random.randn(size, 20)
        f = lambda: mean(X)
        mu, sigma = time(f)
        fast_time.append((size, mu, sigma))

        f = lambda: mean_naive(X)
        mu, sigma = time(f)
        slow_time.append((size, mu, sigma))

    fast_time = np.array(fast_time)
    slow_time = np.array(slow_time)
    fig, ax = plt.subplots()
    ax.errorbar(fast_time[:, 0], fast_time[:, 1], fast_time[:, 2], label='fast mean', linewidth=2)
    ax.errorbar(slow_time[:, 0], slow_time[:, 1], slow_time[:, 2], label='naive mean', linewidth=2)
    ax.set_xlabel('size of dataset')
    ax.set_ylabel('running time')
    plt.legend()
    plt.show()

#do_faces()
do_time()