#!/usr/bin/env python3

import random, util, collections
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('hw10_submission')

############################################################
# Problem 1

def test1a():
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take')
    ]
    for gold, mdp, state, action in tests:
        if not grader.requireIsEqual(gold,
                                     mdp.succAndProbReward(state, action)):
            print(('   state: {}, action: {}'.format(state, action)))
grader.addBasicPart('1a-basic', test1a, 5, description="Basic test for succAndProbReward() that covers several edge cases.")

def test1aHidden():
    mdp = submission.BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                                  threshold=40, peekCost=1)
    startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)
    print(f'startState value is {alg.V[startState]:.2f}')
grader.addHiddenPart('1a-hidden', test1aHidden, 5, description="Hidden test for ValueIteration. Run ValueIteration on BlackjackMDP, then test if V[startState] is correct.")


def test1b():
    mdp = submission.peekingMDP()
    vi = submission.ValueIteration()
    vi.solve(mdp)
    grader.requireIsEqual(mdp.threshold, 20)
    grader.requireIsEqual(mdp.peekCost, 1)
    f = len([a for a in list(vi.pi.values()) if a == 'Peek']) / float(len(list(vi.pi.values())))
    print(f'Percent actions that are \'Peek\' is {f:.4f} ')
    grader.requireIsGreaterThan(.1, f)
    # Feel free to uncomment these lines if you'd like to print out states/actions
    # for k in vi.pi:
    #     print (f'k is {k}, v is {vi.pi[k]}')
grader.addBasicPart('1b-basic', test1b, 4, description="Test for peekingMDP().  Ensure that in at least 10% of states, the optimal policy is to peek.")

############################################################
# Problem 2

def test2a():
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.identityFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback(0, 1, 0, 1)
    grader.requireIsEqual(0, rl.getQ(0, -1))
    grader.requireIsEqual(0, rl.getQ(0, 1))

    rl.incorporateFeedback(1, 1, 1, 2)
    grader.requireIsEqual(0, rl.getQ(0, -1))
    grader.requireIsEqual(0, rl.getQ(0, 1))
    grader.requireIsEqual(0, rl.getQ(1, -1))
    grader.requireIsEqual(1, rl.getQ(1, 1))

    rl.incorporateFeedback(2, -1, 1, 1)
    grader.requireIsEqual(1.9, rl.getQ(2, -1))
    grader.requireIsEqual(0, rl.getQ(2, 1))

grader.addBasicPart('2a-basic', test2a, 5, maxSeconds=10, description="Basic test for incorporateFeedback() using NumberLineMDP.")


def test2aHidden():
    smallMDP = submission.BlackjackMDP(cardValues=[1,5], multiplicity=2, threshold=10, peekCost=1)
    mdp = smallMDP
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                   submission.identityFeatureExtractor,
                                   0.2)
    util.simulate(mdp, rl, 30000)

grader.addHiddenPart('2a-hidden', test2aHidden, 3, maxSeconds=3, description="Hidden test for incorporateFeedback(). Run QLearningAlgorithm on smallMDP, then ensure that getQ returns reasonable value.")

grader.addManualPart('2b', 4, description="Written question: policy comparison for Q-learning vs. value iteration")

# NOTE: this is not a true "test" for grading purposes -- it's worth zero points.  This function exists to help you
# as you're working on question 2b; this question requires a written response on the assignment, but you will need
# to run some code to get the stats that will go into your answer.  Check out the partial implementation of the
# 'simulate_QL_over_MDP' function in submission.py to see one place where you might consider printing these stats.
def run2bHelper():
    featureExtractor = submission.identityFeatureExtractor
    submission.simulate_QL_over_MDP(submission.smallMDP, featureExtractor)
    submission.simulate_QL_over_MDP(submission.largeMDP, featureExtractor)
grader.addBasicPart('2b-helper', run2bHelper, 0, maxSeconds=60, description="Helper function to run Q-learning simulations for question 2b.")

def test2c():
    mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.blackjackFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback((7, None, (0, 1)), 'Quit', 7, (7, None, None))
    grader.requireIsEqual(28, rl.getQ((7, None, (0, 1)), 'Quit'))
    grader.requireIsEqual(7, rl.getQ((7, None, (1, 0)), 'Quit'))
    grader.requireIsEqual(14, rl.getQ((2, None, (0, 2)), 'Quit'))
    grader.requireIsEqual(0, rl.getQ((2, None, (0, 2)), 'Take'))
grader.addBasicPart('2c-basic', test2c, 5, maxSeconds=10, description="Basic test for blackjackFeatureExtractor.  Runs QLearningAlgorithm using blackjackFeatureExtractor, then checks to see that Q-values are correct.")

# grader.addManualPart('2d', 4, description="Written question: reward comparison for applying policy to baseline and modified MDP")

# NOTE: as in 2b above, this is not a real test -- just a helper function to run some code
# to produce stats that will allow you to answer written question 2d.
# def run2dHelper():
#     submission.compare_changed_MDP(submission.originalMDP, submission.newThresholdMDP, submission.blackjackFeatureExtractor)
# grader.addBasicPart('2d-helper', run2dHelper, 0, maxSeconds=60, description="Helper function to compare rewards when simulating RL over two different MDPs in question 2d.")

grader.grade()
