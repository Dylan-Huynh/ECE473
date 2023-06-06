#!/usr/bin/python

import random
import collections
import math
import sys
from util import *


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    features = {}
    y = x.split()
    for i in y:
        if i not in features:
            features[i] = 1
        else:
            features[i] += 1
    return features
    # END_YOUR_CODE


############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!  You should call
    evaluatePredictor() on both trainExamples and testExamples to see
    how you're doing as you learn after each iteration, storing the
    results as required for the resulting print statement included below.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)

    for i in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            if dotProduct(weights, phi) * y < 1:
                for f, v in list(phi.items()):
                    weights[f] = weights.get(f, 0) + v * eta * y
        '''trainError = evaluatePredictor(trainExamples,
                                       lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        devError = evaluatePredictor(testExamples,
                                     lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print(("Official: train error = %s, dev error = %s" % (trainError, devError))) '''

    # END_YOUR_CODE

    return weights


############################################################
# Problem 2c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
        for x in weights.keys():
            num = random.randint(0, 42)
            if num > 0:
                phi[x] = num
        if dotProduct(phi, weights) > 1:
            y = 1
        else:
            0
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 2e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''

    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        word = "".join(x.split())
        dic = {}
        for i in range(len(word) + 1 - n):
            ngram = word[i:i + n]
            if i not in dic:
                dic[ngram] = 1
            else:
                dic[ngram] += 1
        return dic
        # END_YOUR_CODE

    return extract
