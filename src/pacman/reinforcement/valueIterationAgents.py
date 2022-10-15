# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        states = mdp.getStates()
        # This is my value iteration loop and this will run as many iterations as needed
        # I will change values through the temp variable then set my values variable to that
        # I do this so when values are changed they won't affect other calculations too early
        # Because their U(s`) value would be changed during the iteration instead of after
        for i in range(iterations):
            temp = util.Counter()
            # Every iteration we need to compute every state
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                # This handles the terminal case where we have no actions
                if self.mdp.isTerminal(state):
                    temp[state] = 0
                else:
                    maxV = float("-inf")
                    # In every state we need to compute every action
                    for action in actions:
                        # This uses our qValue method for calculations
                        qValue = self.computeQValueFromValues(state, action)
                        # Keeping track of the max value for all action in our state
                        maxV = max(maxV, qValue)
                        temp[state] = maxV
            # Setting the values for this iteration
            self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # This will perform the QValue computation for a state action pair
        qValue = 0
        # The loop for all possible next states by using the transition function
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += prob * (self.mdp.getReward(state, action, nextState)
                              + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        # Handling the terminal case
        if self.mdp.isTerminal(state):
            return None
        value = float("-inf")
        bestAction = None
        # I iterate through the actions and find the best
        for action in actions:
            temp = self.computeQValueFromValues(state, action)
            # Here I use this if statement instead of the max() function to keep the maximum value
            # I do this so I can set the right bestAction when there's a better value
            # Because that is what I need to return
            if temp >= value:
                value = temp
                bestAction = action
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
