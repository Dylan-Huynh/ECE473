import hw4_submission as train_submission
import util
from collections import defaultdict

# Read in examples
trainExamples = util.readExamples('names.train')
devExamples = util.readExamples('names.dev') #validation set

def featureExtractor(x):  # phi(x)
    # example x == the Lesser Antilles and
    phi = defaultdict(float)
    words = x.split()
    leftContext, entity, rightContext = words[0], words[1:-1], words[-1]
    
    # Problem 3a: add this feature template
    phi['entity is ' + ' '.join(entity)] = 1

    # Problem 3b: add these feature templates
    # phi['left is ' + leftContext] = 1
    # phi['right is ' + rightContext] = 1

    for word in entity:
        # Problem 3c: add these feature templates
        # phi['entity contains ' + word] = 1

        # Problem 3d: add these feature templates
        # phi['entity contains suffix ' + word[-4:]] = 1
        # phi['entity contains prefix ' + word[:4]] = 1
        None

    return phi

# Learn a predictor
weights = train_submission.learnPredictor(trainExamples, devExamples, featureExtractor, 20, 0.1)
util.outputWeights(weights, 'weights')
util.outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')

# Test!!!
testExamples = util.readExamples('names.test')
predictor = lambda x : 1 if util.dotProduct(featureExtractor(x), weights) > 0 else -1
print('test error =', util.evaluatePredictor(testExamples, predictor))

# predictor = lambda x : -1
# print('Always -1 scores on dev data: ', util.evaluatePredictor(devExamples, predictor))
