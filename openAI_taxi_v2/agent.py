import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.epsilon = 0.05
        self.alpha = 0.5
        self.gamma = 1
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.epsilon_greedy_policy(self.Q, state, self.epsilon)[0]

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action_egp, probs = self.epsilon_greedy_policy(self.Q, next_state, self.epsilon)
        old_Q = self.Q[state][action]    
        self.Q[state][action] = old_Q + self.alpha*(reward+(self.gamma*np.dot(probs, self.Q[next_state]))-old_Q)
        
    def epsilon_greedy_policy(self, Q, state, epsilon):
        if (Q[state] == Q[state][0]).all():
            probs = [1/self.nA]*self.nA
        else:
            prob_greedy_action = 1-epsilon+(epsilon/self.nA)
            prob_non_greedy_action = epsilon/self.nA
            probs = np.array([prob_non_greedy_action for i in Q[state]])
            probs[np.argmax(Q[state])] = prob_greedy_action

        action = np.random.choice(np.arange(self.nA), p=probs)
        return (action, probs)