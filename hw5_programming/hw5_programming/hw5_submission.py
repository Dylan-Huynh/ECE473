#!/usr/bin/python

import numpy as np
import math
from util import *

############################################################
# Problem 1: k-means
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    assignments = [-1] * len(examples)
    centroids = {}
    loss = 0
    distances = [0] * len(examples)
    prev_distances = [1] * len(examples)
    centroids = [examples[i] for i in range(K)]
    x_magnitude = [dotProduct(i, i) for i in examples]

    for t in range(maxIters):
        if np.array_equal(distances, prev_distances):
            break
        prev_distances = np.copy(distances)
        loss = 0

        y_magnitude = [dotProduct(i, i) for i in centroids]

        for i in range(len(examples)):
            current_min = math.inf
            for j in range(K):
                distance = abs(x_magnitude[i] - 2 * dotProduct(examples[i], centroids[j]) + y_magnitude[j])
                if distance < current_min:
                    current_min = distance
                    distances[i] = current_min
                    assignments[i] = j
        loss = sum(distances)
        for j in range(K):
            cluster_sum = {}
            count = 0
            for i in range(len(examples)):
                if assignments[i] == j:
                    for k, l in examples[i].items():
                        cluster_sum[k] = cluster_sum.get(k, 0) + l
                    count += 1
            centroids[j] = {m: n / count for m, n in cluster_sum.items()}

    return centroids, assignments, loss
    # END_YOUR_CODE
