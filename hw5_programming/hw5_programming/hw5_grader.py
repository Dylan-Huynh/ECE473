#!/usr/bin/python3

import graderUtil
import util
import time
from util import *

grader = graderUtil.Grader()
submission = grader.load('hw5_submission')

############################################################
# Problem 1: clustering
############################################################

grader.addManualPart('1a', maxPoints=5, description='simulating 2-means')

# basic test for k-means
def test4b0():
    random.seed(42)
    x1 = {0:0, 1:0}
    x2 = {0:0, 1:1}
    x3 = {0:0, 1:2}
    x4 = {0:0, 1:3}
    x5 = {0:0, 1:4}
    x6 = {0:0, 1:5}
    examples = [x1, x2, x3, x4, x5, x6]
    centers, assignments, totalCost = submission.kmeans(examples, 2, maxIters=10)
    # (there are two stable centroid locations)
    grader.requireIsEqual(True, round(totalCost, 3)==4 or round(totalCost, 3)==5.5 or round(totalCost, 3)==5.0)
grader.addBasicPart('1b-0-basic', test4b0, maxSeconds=1, description="test basic k-means on hardcoded datapoints")

'''def test4b1():
    random.seed(42)
    K = 6
    bestCenters = None
    bestAssignments = None
    bestTotalCost = None
    examples = generateClusteringExamples(numExamples=1000, numWordsPerTopic=3, numFillerWords=1000)
    centers, assignments, totalCost = submission.kmeans(examples, K, maxIters=100)
grader.addHiddenPart('1b-1-hidden', test4b1, maxPoints=1, maxSeconds=3, description="test stability of cluster assignments")

def test4b2():
    random.seed(42)
    K = 6
    bestCenters = None
    bestAssignments = None
    bestTotalCost = None
    examples = generateClusteringExamples(numExamples=1000, numWordsPerTopic=3, numFillerWords=1000)
    centers, assignments, totalCost = submission.kmeans(examples, K, maxIters=100)
grader.addHiddenPart('1b-2-hidden', test4b2, maxPoints=1, maxSeconds=3, description="test stability of cluster locations")

def test4b3():
    random.seed(42)
    K = 6
    bestCenters = None
    bestAssignments = None
    bestTotalCost = None
    examples = generateClusteringExamples(numExamples=10000, numWordsPerTopic=3, numFillerWords=10000)
    centers, assignments, totalCost = submission.kmeans(examples, K, maxIters=100)
    grader.requireIsLessThan(10e6, totalCost)
grader.addHiddenPart('1b-3-hidden', test4b3, maxPoints=2, maxSeconds=4, description="make sure the code runs fast enough")

grader.addManualPart('1c', maxPoints=5, description='handling same-cluster constraints')
grader.addManualPart('1d', maxPoints=5, description='rerunning kmeans')
grader.addManualPart('1e', maxPoints=5, description='scaling kmeans')'''

grader.grade()
