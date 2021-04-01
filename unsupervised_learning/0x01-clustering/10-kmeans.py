#!/usr/bin/env python3
"""K-means with sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    centroids, clss, w = sklearn.cluster.k_means(X, k)
    return centroids, clss
