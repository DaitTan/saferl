#!/usr/bin/env python3
import random

import hydra

import util

from learningAgents import ReinforcementAgent

class LNNQAgent(ReinforcementAgent):
    def __init__(
        self,
        lnn="our_lnn.LNNShielding",
        extractor="featureExtractors.CloseGhostExtractor",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.index = 0  # This is always Pacman
        self.values = util.Counter()

        lnn_cfg = {'_target_': lnn}
        self.lnn = hydra.utils.instantiate(lnn_cfg)

        extractor_cfg = {'_target_': extractor}
        self.feature_extractor = hydra.utils.instantiate(extractor_cfg)

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

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # This is very similar to computeValueFromQValues
        q_values = util.Counter()
        actions = self.getLegalActions(state)
        # When we have no actions we are in a terminal state
        if len(actions) == 0:
            return None
        # Getting the q_values for every action
        for action in actions:
            q_values[action] = self.getQValue(state, action)
        
        # Sort the q_values by values and return the keys
        best_actions = [k[0] for k in sorted(q_values.items(), reverse=True, key=lambda x: x[1])]
        for action in best_actions:
            close_ghosts = self.feature_extractor.getFeatures(state, action)
            if self.lnn.is_action_safe(close_ghosts):
                # print(f'I am the action chosen: {action}')
                # print(f'I am the q_values: {q_values}')
                # print(f'I am the best actions: {best_actions}')
                return action
        print('Every action lead to a contradiction!')
        # If we have reached this point then every valid action was deemed
        # unsafe by the LNN, so just pick the action with the best Q-Value
        return best_actions[0]

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

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if state is None:
            print('STATE IS NONEIN GET ACTION')
        # If we are in a terminal state return None
        if len(legalActions) == 0:
            return None
        # If the flip is successful under probability e
        # Then we should explore by choosing a random move
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        # If the coin flipped failed then return our current best action
        action = self.computeActionFromQValues(state)
        self.doAction(state, action)
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

    def final(self, state):
        super().final(state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print('WE MADE IT TO THE END OF TRAINING')

class LNNApproximateQAgent(LNNQAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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