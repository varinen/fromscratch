import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import sklearn.metrics

iris = datasets.load_iris()
print('x shape is {}'.format(iris.data.shape))
print('y shape is {}'.format(iris.target.shape))


# ===YOU SHOULD EDIT THIS FUNCTION===
def pairwise_distance_matrix(X, Y):
    distance_matrix = sklearn.metrics.pairwise.euclidean_distances(X, Y)
    return distance_matrix;


def KNN(k, X, y, Xtest):
    """K nearest neighbors
    Arguments
    ---------
    k: int using k nearest neighbors.
    X: the training set
    y: the classes
    Xtest: the test set which we want to predict the classes

    Returns
    -------
    ypred: predicted classes for Xtest

    """
    N, D = X.shape
    M, _ = Xtest.shape
    num_classes = len(np.unique(y))

    # 1. Compute distance with all flowers
    distance = np.zeros((N, M))  # EDIT THIS to use "pairwise_distance_matrix"
    distance = pairwise_distance_matrix(X, Xtest)

    # 2. Find indices for the k closest flowers
    idx = np.argsort(distance.T, axis=1)[:, :K]

    # 3. Vote for the major class
    ypred = np.zeros((M, num_classes))

    for m in range(M):
        klasses = y[idx[m]]
        for k in np.unique(klasses):
            ypred[m, k] = len(klasses[klasses == k]) / K

    return np.argmax(ypred, axis=1)


K = 3

X = iris.data[:, :2]  # use first two features for simplicity
y = iris.target

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

ypred = []
data = np.array([xx.ravel(), yy.ravel()]).T
ypred = KNN(K, X, y, data)

fig, ax = plt.subplots(figsize=(4, 4))

