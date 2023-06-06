#!/usr/bin/env python3

import random, util, collections
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('hw9_submission')

############################################################
# Problem 1

grader.addManualPart('1a', 3, description="Written question: value iteration in basic MDP")
grader.addManualPart('1b', 3, description="Written question: optimal policy in basic MDP")

############################################################
# Problem 2

class AddNoiseMDP(util.MDP):
    def __init__(self, originalMDP):
        self.originalMDP = originalMDP

    def startState(self):
        return self.originalMDP.startState()

    # Return set of actions possible from |state|.
    def actions(self, state):
        return self.originalMDP.actions(state)

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        originalSuccAndProbReward = self.originalMDP.succAndProbReward(state, action)
        newSuccAndProbReward = []
        for state, prob, reward in originalSuccAndProbReward:
            newProb = 0.5 * prob + 0.5 / len(originalSuccAndProbReward)
            newSuccAndProbReward.append((state, newProb, reward))
        return newSuccAndProbReward

    # Return set of actions possible from |state|.
    def discount(self):
        return self.originalMDP.discount()

def test2a():
    mdp = submission.CounterexampleMDP()
    mdp.computeStates()
    algorithm = submission.ValueIteration()
    algorithm.solve(mdp, .001)
    originalVal = algorithm.V[mdp.startState()]
    mdp = AddNoiseMDP(mdp)
    mdp.computeStates()
    algorithm.solve(mdp, .001)
    newVal = algorithm.V[mdp.startState()]
    print("Original value: {}, New value: {}".format(originalVal, newVal))
grader.addHiddenPart('2a-hidden', test2a, 3, description="Hidden test for CounterexampleMDP. Ensure that V[startState] is greater after noise is added.")
grader.addManualPart('2a-written', 3, description="Written question: describe your counter-example")

grader.addManualPart('2b', 3, description="Written question: single-pass algorithm for node values in acyclic MDP")
grader.addManualPart('2c', 3, description="Written question: define new MDP solver for discounts < 1")

grader.grade()
