#!/usr/bin/python3

import graderUtil
import util
import time
from util import *

grader = graderUtil.Grader()
submission = grader.load('hw4_submission')

############################################################
# Problem 1: building intuition
############################################################

grader.addManualPart('1a', maxPoints=2, description='simulate SGD')
grader.addManualPart('1b', maxPoints=2, description='create small dataset')

############################################################
# Problem 2: sentiment classification
############################################################

### 2a

# Basic sanity check for feature extraction
def test2a0():
    ans = {"a":2, "b":1}
    grader.requireIsEqual(ans, submission.extractWordFeatures("a b a"))
grader.addBasicPart('2a-0-basic', test2a0, maxSeconds=1, description="basic test")

def test2a1():
    random.seed(42)
    for i in range(10):
        sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
    submission_ans = submission.extractWordFeatures(sentence)
grader.addHiddenPart('2a-1-hidden', test2a1, maxSeconds=1, description="test multiple instances of the same word in a sentence")

### 2b

def test2b0():
    trainExamples = (("hello world", 1), ("goodnight moon", -1))
    testExamples = (("hello", 1), ("moon", -1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    grader.requireIsGreaterThan(0, weights["hello"])
    grader.requireIsLessThan(0, weights["moon"])
grader.addBasicPart('2b-0-basic', test2b0, maxSeconds=1, description="basic sanity check for learning correct weights on two training and testing examples each")

def test2b1():
    trainExamples = (("hi bye", 1), ("hi hi", -1))
    testExamples = (("hi", -1), ("bye", 1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    grader.requireIsLessThan(0, weights["hi"])
    grader.requireIsGreaterThan(0, weights["bye"])
grader.addBasicPart('2b-1-basic', test2b1, maxSeconds=1, description="test correct overriding of positive weight due to one negative instance with repeated words")

def test2b2():
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, dev error = %s" % (trainError, devError)))
    grader.requireIsLessThan(0.04, trainError)
    grader.requireIsLessThan(0.30, devError)
grader.addBasicPart('2b-2-basic', test2b2, maxPoints=2, maxSeconds=8, description="test classifier on real polarity dev dataset")

### 2c

def test2c0():
    weights = {"hello":1, "world":1}
    data = submission.generateDataset(5, weights)
    for datapt in data:
        grader.requireIsEqual((util.dotProduct(datapt[0], weights) >= 0), (datapt[1] == 1))
grader.addBasicPart('2c-0-basic', test2c0, maxSeconds=1, description="test correct generation of random dataset labels")

def test2c1():
    weights = {}
    for i in range(100):
        weights[str(i + 0.1)] = 1
    data = submission.generateDataset(100, weights)
    for datapt in data:
        grader.requireIsEqual(False, dotProduct(datapt[0], weights) == 0)
grader.addBasicPart('2c-1-basic', test2c1, maxSeconds=1, description="test that the randomly generated example actually coincides with the given weights")

### 2d

grader.addManualPart('2d', maxPoints=2, description='error analysis')

### 2e

def test2e0():
    fe = submission.extractCharacterFeatures(3)
    sentence = "hello world"
    ans = {"hel":1, "ell":1, "llo":1, "low":1, "owo":1, "wor":1, "orl":1, "rld":1}
    grader.requireIsEqual(ans, fe(sentence))
grader.addBasicPart('2e-0-basic', test2e0, maxSeconds=1, description="test basic character n-gram features")

def test2e1():
    random.seed(42)
    for i in range(10):
        sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
        for n in range(1, 4):
            submission_ans = submission.extractCharacterFeatures(n)(sentence)
grader.addHiddenPart('2e-1-hidden', test2e1, maxSeconds=1, description="test feature extraction on repeated character n-grams")

### 2f

grader.addManualPart('2f', maxPoints=3, description='explain value of n-grams')

grader.grade()
