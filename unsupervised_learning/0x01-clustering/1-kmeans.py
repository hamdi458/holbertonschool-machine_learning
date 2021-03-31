#!/usr/bin/env python3
""" K-means """
import numpy as np


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each
    point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def kmeans(X, k, iterations=1000):
    """ k means algo """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    maxi = np.max(X, axis=0)
    mini = np.min(X, axis=0)
    centroid = np.random.uniform(low=mini, high=maxi, size=(k, X.shape[1]))

    for point in range(iterations):
        copy_centroid = np.copy(centroid)
        clss = closest_centroid(X, centroid)
        for i in range(k):
            index = np.transpose(np.nonzero(clss == i))
            if index.shape[0] == 0:
                centroid[i] = np.random.uniform(maxi, mini, (1, X.shape[1]))
            else:
                idxX = np.take(X, index, axis=0)
                centroid[i] = (idxX.mean(axis=0))
        clss = closest_centroid(X, centroid)
        if (copy_centroid == centroid).all():
            return centroid, clss
    return centroid, clss
