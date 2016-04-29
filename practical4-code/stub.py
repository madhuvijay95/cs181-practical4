

# Imports.
import numpy as np
import numpy.random as npr
import cPickle as pickle
import matplotlib.pyplot as plt
import sys

from SwingyMonkey import SwingyMonkey


class Learner(object):
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None
        self.init_val = 5
        self.learning_rate = lambda t : 0.2
        self.discount = 1
        self.Q = dict()
        self.epsilon = lambda t : t ** (-1)
        self.t = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    # Time required for an object with a specified vertical velocity and acceleration to move upward by a certain
    # height. (This is derived using the simple kinematic equation h = v*t + 1/2*a*t^2, along with the quadratic
    # equation.) The function returns np.nan if there is no solution.
    def __dist_convert(self, velocity, acceleration, height):
        # return the largest root of two roots of h = v*t + 1/2*a*t^2
        t1 = float(-velocity + np.sqrt(velocity**2 + 2*acceleration*height)) / acceleration
        t2 = float(-velocity - np.sqrt(velocity**2 + 2*acceleration*height)) / acceleration
        return max(t1, t2)

    # Generate a tuple representing a state, using the output of SwingyMonkey.get_state().
    def __state_convert(self, state):
        # distances measuring the vertical distance from the monkey to the top and bottom of the tree
        bottom_diff = state['tree']['bot'] - state['monkey']['bot']
        top_diff = state['tree']['top'] - state['monkey']['top']
        # monkey's velocity
        velocity = state['monkey']['vel']
        # compute the remaining time until the monkey reaches the vertical height of the bottom and top of the tree
        bottom_tree_time = self.__dist_convert(velocity, -self.gravity, bottom_diff)
        top_tree_time = self.__dist_convert(velocity, -self.gravity, top_diff)
        # compute the remaining time until the monkey reaches the vertical height of the bottom of the screen
        bottom_time = self.__dist_convert(velocity, -self.gravity, -state['monkey']['bot'])
        # function that rounds to the nearest integer, while casting NaN's as sys.maxint (the highest storable int)
        round_nan = lambda x : round(x) if not np.isnan(x) else sys.maxint

        # The following code shows an example of a more straightforward way to encode the state space, by simply using
        # the level of gravity, the distance to the next, tree, the height of the tree, the velocity of the monkey, and
        # the height of the monkey (all of which were rounded sufficiently to compress the state space).
        # As explained in the write-up, this approach was less successful than the un-commented method.
        #return self.gravity, round_nan(0.1*state['tree']['dist']), round_nan(0.02*state['tree']['top']), \
        #       round_nan(0.2*state['monkey']['vel']), round_nan(0.02*state['monkey']['top'])

        # Represent a state as: (1) gravity strength, (2) bottom_tree_time (rounded to the nearest multiple of 4);
        # (2) top_tree_time (rounded to the nearest multiple of 4); (3) bottom_time (rounded to the nearest multiple
        # of 4); and (4) the distance to the tree (rounded to the nearest multiple of 10).
        return self.gravity, round_nan(0.25*bottom_tree_time), round_nan(0.25*top_tree_time), \
               round_nan(0.25*bottom_time), round_nan(0.1*state['tree']['dist'])

    # Function that uses Q-learning based on the current and past state, and returns action decisions.
    def action_callback(self, state):

        # increment the time step
        self.t += 1

        # compute the gravity of the current game, if it hasn't been stored yet
        if self.gravity is None and self.last_state is not None:
            if state['monkey']['vel'] < self.last_state['monkey']['vel']:
                self.gravity = self.last_state['monkey']['vel'] - state['monkey']['vel']

        if self.gravity is not None:
            # use the __state_convert function to construct representations of the previous and current states
            state_rep_old = self.__state_convert(self.last_state)
            state_rep_new = self.__state_convert(state)

            # initialize any relevant missing values in the Q table
            if (state_rep_old, 0) not in self.Q:
                self.Q[(state_rep_old, 0)] = self.init_val
            if (state_rep_old, 1) not in self.Q:
                self.Q[(state_rep_old, 1)] = self.init_val
            if (state_rep_new, 0) not in self.Q:
                self.Q[(state_rep_new, 0)] = self.init_val
            if (state_rep_new, 1) not in self.Q:
                self.Q[(state_rep_new, 1)] = self.init_val

            # take a random action with probability epsilon_t
            if npr.rand() < self.epsilon(self.t):
                new_action = npr.choice([0,1])
            # otherwise, choose the action that has a higher expected value in the Q dictionary
            else:
                new_action = np.argmax([self.Q[(state_rep_new, 0)], self.Q[(state_rep_new, 1)]])

            # temporal difference error (using the Q-learning formulation, rather than simple SARSA)
            td = self.last_reward + self.discount * max(self.Q[(state_rep_new, 0)], self.Q[(state_rep_new, 1)])\
                 - self.Q[(state_rep_old, int(self.last_action))]
            # update the Q dictionary value using the learning rate alpha_t and the temporal difference error td
            self.Q[(state_rep_old, int(self.last_action))] += self.learning_rate(self.t) * td
        else:
            new_action = npr.choice([0,1])

        self.last_action = new_action
        self.last_state  = state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


# Have the agent play a sequence of games in order to learn how to play well (using Q-learning).
def run_games(learner, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    scores = []
    scores1 = []
    scores4 = []
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        scores.append(swing.score)
        if swing.gravity == 1:
            scores1.append(swing.score)
        elif swing.gravity == 4:
            scores4.append(swing.score)
        # Reset the state of the learner.
        learner.reset()
        
    # plot the game scores over time, and save to a png
    plt.plot(range(iters), scores)
    plt.title('Scores')
    plt.get_current_fig_manager().window.showMaximized()
    plt.savefig('scores.png')
    plt.show()
    plt.close()

    window = 50
    # compute a moving average of the score
    ma = np.convolve(scores, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(len(ma)) + window, ma)
    plt.title('50-Game Moving Average Score')
    plt.get_current_fig_manager().window.showMaximized()
    plt.savefig('scores_ma.png')
    plt.show()
    plt.close()

    print 'When gravity=1: %d games, with an average score of %.3f' % (len(scores1), np.mean(scores1))
    print 'When gravity=4: %d games, with an average score of %.3f' % (len(scores4), np.mean(scores4))
    print 'For all games: %d games, with an average score of %.3f' % (len(scores), np.mean(scores))
    # store the scores in a pickle file
    pickle.dump((scores, scores1, scores4), open('scores.p', 'w'))
    return


# Simulate some games at a lower speed in order to demonstrate how the trained agent plays.
def sim_games(learner, iters = None, t_len=50):
    i = 0
    # demonstrate the learner playing the game; this will end only when you close the window manually
    while (iters == None or i < iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,
                             tick_length=50,
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        learner.reset()
        i += 1
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games to train the agent.
    run_games(agent, iters=500, t_len=0)

    # Play 3 sample games at a lower speed to demonstrate the agent.
    sim_games(agent, iters=3, t_len=50)