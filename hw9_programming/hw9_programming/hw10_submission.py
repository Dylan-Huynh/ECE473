import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 1a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 37 lines of code, but don't worry if you deviate from this)
        result = []
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state
        if deckCardCounts is None:
            return []
        elif action == 'Quit' or sum(deckCardCounts) == 0:
            nextState = (0, None, None)
            nextReward = totalCardValueInHand
            if totalCardValueInHand > self.threshold:
                nextReward = 0
            result.append((nextState, 1.0, nextReward))
        elif action == 'Take':
            if nextCardIndexIfPeeked is not None:
                deckCardCountsList = list(deckCardCounts)
                deckCardCountsList[nextCardIndexIfPeeked] -= 1
                newValueInHand = self.cardValues[nextCardIndexIfPeeked] + totalCardValueInHand
                if newValueInHand > self.threshold:
                    nextState = (newValueInHand, None, None)
                elif sum(deckCardCountsList) == 0:
                    nextState = (newValueInHand, None, None)
                    nextReward = newValueInHand
                else:
                    nextState = (newValueInHand, None, tuple(deckCardCountsList))
                result.append((nextState, 1.0, 0))
            else:
                for index, item in enumerate(deckCardCounts):
                    if item > 0:
                        deckCardCountsList = list(deckCardCounts)
                        nextProb = float(item) / sum(deckCardCounts)
                        deckCardCountsList[index] -= 1
                        newValueInHand = self.cardValues[index] + totalCardValueInHand
                        nextReward = 0
                        if newValueInHand > self.threshold:
                            nextState = (newValueInHand, None, None)
                        elif sum(deckCardCountsList) == 0:
                            nextState = (newValueInHand, None, None)
                            nextReward = newValueInHand
                        else:
                            nextState = (newValueInHand, None, tuple(deckCardCountsList))
                        result.append((nextState, nextProb, nextReward))
        elif action == 'Peek':
            if nextCardIndexIfPeeked is not None:
                return []
            for index, item in enumerate(deckCardCounts):
                if item > 0:
                    nextProb = float(item) / sum(deckCardCounts)
                    nextState = (totalCardValueInHand, index, deckCardCounts)
                    nextReward = -self.peekCost
                    result.append((nextState, nextProb, nextReward))
        return result
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 1b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    mdp1 = BlackjackMDP(cardValues=[1, 5, 15], multiplicity=10,
                        threshold=20, peekCost=1)
    return mdp1

    # END_YOUR_CODE

############################################################
# Problem 2a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a dict of {feature name => feature value}.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f,v in self.featureExtractor(state, action).items():
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        V_opt = 0.0
        if newState is not None:
            V_opt = max([self.getQ(newState, newAction) for newAction in self.actions(newState)])
        Q_opt = self.getQ(state, action)
        adjustment = -self.getStepSize() * (Q_opt - (reward + self.discount * V_opt))
        for item in self.featureExtractor(state, action):
            key, value = item
            self.weights[key] = self.weights[key] + adjustment * value
        # END_YOUR_CODE

# Return a single-element dict containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return {featureKey: featureValue}

############################################################
# Problem 2b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 2b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    valueIteration = ValueIteration()
    valueIteration.solve(mdp)
    vi_pi = valueIteration.pi
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
    util.simulate(mdp, rl, numTrials=30000, verbose=False)
    rl.explorationProb = 0
    diff, total = 0, len(mdp.states)
    for state in mdp.states:
        if vi_pi[state] != rl.getAction(state):
            diff += 1
    print('{:.3f}'.format(100 * diff / total) + '%')
    # END_YOUR_CODE

############################################################
# Problem 2c: features for Q-learning.

# You should return a dict of {feature key => feature value}.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the dict you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck. (1 feature)
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- For each face value, add an indicator for the action and the number of cards remaining with that face value (len(counts) features)
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    features = []
    featureKey = (action, total)
    featureValue = 1
    features.append((featureKey, featureValue))
    if counts is not None:
        countsList = list(counts)
        for index, item in enumerate(counts):
            featureKey = (action, index, item)
            featureValue = 1
            features.append((featureKey, featureValue))
            if item > 0:
                countsList[index] = 1
        featureKey = (action, tuple(countsList))
        featureValue = 1
        features.append((featureKey, featureValue))
    return features
    # END_YOUR_CODE

############################################################
# Problem 2d: What happens when the MDP changes underneath you?!
#    -- Spring 2022 semester: this problem is provided for your entertainment and enrichment.  Do not turn anything in.

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 2b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 2d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    original_mdp.computeStates()
    vi = ValueIteration()
    vi.solve(originalMDP)

    rl = util.FixedRLAlgorithm(vi.pi.copy())
    rewards = util.simulate(modified_mdp, rl, numTrials=10000, maxIterations=1000, verbose=False, sort=False)
    rl.explorationProb = 0.0
    # print(rewards)
    modified_mdp.computeStates()
    rl = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, 0.2)

    rewards = util.simulate(modified_mdp, rl, numTrials=10000, maxIterations=1000, verbose=False, sort=False)
    # END_YOUR_CODE

