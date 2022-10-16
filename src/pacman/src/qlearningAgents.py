# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # This will store our values
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # If we haven't evaluated the state set its value to 0
        if (state, action) not in self.values:
            self.values[(state, action)] = 0.0
        # Otherwise return the value
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # This will store the qValues for all the actions in this state
        qValues = util.Counter()
        actions = self.getLegalActions(state)
        # This is the terminal state case where their value = 0
        if len(actions) == 0:
            return 0.0
        # Finding the max value of all actions
        for action in actions:
            qValues[action] = self.getQValue(state, action)
        # This will return the value of the maximum action
        return qValues[qValues.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # This is very similar to computeValueFromQValues
        qValues = util.Counter()
        actions = self.getLegalActions(state)
        # When we have no actions we are in a terminal state
        if len(actions) == 0:
            return None
        # Getting the QValues for every action
        for action in actions:
            qValues[action] = self.getQValue(state, action)
        # This part is to have the agent choose a random action from the best choices
        # This will help with the exploration part
        bestValue = self.computeValueFromQValues(state) # This method will return the highest value in that state
        # If the value of that action equals out best value add it to our best actions
        bestActions = [action for action in actions if qValues[action] == bestValue]
        # If we only have one best action that one will still be chosen otherwise it's random
        # Among the ties for our best action
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        # If we are in a terminal state return None
        if len(legalActions) == 0:
            return None
        # If the flip is successful under probability e
        # Then we should explore by choosing a random move
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        # If the coin flipped failed then return our current best action
        action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # This is our Q(s, a)
        currentValue = self.values[(state, action)]
        # This is our Q(s', a')
        nextValue = self.computeValueFromQValues(nextState)
        # This is the application to the bellman equation
        learning = reward + self.discount * nextValue - currentValue
        # This is the full Q-Learning equation
        updated = currentValue + self.alpha * learning
        self.values[(state, action)] = updated

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # This returns our features
        features = self.featExtractor.getFeatures(state, action)
        value = 0
        # This loop represents the sigma in the approximate Q-Learning equation
        for feat in features:
            # The feature value * the weight of that feature
            value += features[feat] * self.getWeights()[feat]
        return value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # This is our Q(s, a)
        currentValue = self.getQValue(state, action)
        # This is our Q(s', a')
        nextValue = self.computeValueFromQValues(nextState)
        difference = reward + self.discount * nextValue - currentValue
        # We must iterate through all of the features to update each weight for them
        # This weight is the value of the feature at our state
        features = self.featExtractor.getFeatures(state, action)
        for feat in features:
            self.weights[feat] = self.weights[feat] + self.alpha * difference * features[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
