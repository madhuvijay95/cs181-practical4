# Imports.
import numpy as np
import numpy.random as npr
import cPickle as pickle
import matplotlib.pyplot as plt
import sys

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None
        self.init_val = 5
        self.learning_rate = lambda t : 0.2
        #self.learning_rate = lambda t : t ** (-0.51)
        self.discount = 1
        self.Q = dict()
        self.epsilon = lambda t : t ** (-1)
        self.t = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    def __dist_convert(self, velocity, acceleration, height):
        t1 = float(-velocity + np.sqrt(velocity**2 + 2*acceleration*height)) / acceleration
        t2 = float(-velocity - np.sqrt(velocity**2 + 2*acceleration*height)) / acceleration
        return max(t1, t2)

    def __state_convert(self, state):
        bottom_diff = state['tree']['bot'] - state['monkey']['bot']
        top_diff = state['tree']['top'] - state['monkey']['top']
        velocity = state['monkey']['vel']
        bottom_tree_time = self.__dist_convert(velocity, -self.gravity, bottom_diff)
        #bottom_time = float(velocity + np.sqrt(velocity**2 - 2*self.gravity*bottom_diff)) / self.gravity
        top_tree_time = self.__dist_convert(velocity, -self.gravity, top_diff)
        #top_time = float(velocity + np.sqrt(velocity**2 - 2*self.gravity*top_diff)) / self.gravity
        bottom_time = self.__dist_convert(velocity, -self.gravity, -state['monkey']['bot'])
        round_nan = lambda x : round(x) if not np.isnan(x) else sys.maxint
        return round_nan(0.25*bottom_tree_time), round_nan(0.25*top_tree_time),\
               round_nan(0.25*bottom_time), round_nan(0.1*state['tree']['dist'])

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_state  = state
        self.t += 1

        if self.gravity is None and self.last_state is not None:
            if state['monkey']['vel'] < self.last_state['monkey']['vel']:
                self.gravity = self.last_state['monkey']['vel'] - state['monkey']['vel']
                #print self.gravity

        new_action = npr.rand() < 0.1
        if self.gravity is not None:
            state_rep_old = self.__state_convert(self.last_state)
            state_rep_new = self.__state_convert(new_state)

            # initialize any relevant missing values in the Q table
            if (state_rep_old, 0) not in self.Q:
                self.Q[(state_rep_old, 0)] = self.init_val
            if (state_rep_old, 1) not in self.Q:
                self.Q[(state_rep_old, 1)] = self.init_val
            #if (state_rep_old, int(self.last_action)) not in self.Q:
            #    self.Q[(state_rep_old, int(self.last_action))] = self.init_val
            if (state_rep_new, 0) not in self.Q:
                self.Q[(state_rep_new, 0)] = self.init_val
            if (state_rep_new, 1) not in self.Q:
                self.Q[(state_rep_new, 1)] = self.init_val

            if npr.rand() < self.epsilon(self.t):
                new_action = npr.choice([0, 1])
            else:
                new_action = np.argmax([self.Q[(state_rep_new, 0)], self.Q[(state_rep_new, 1)]])

            td = self.last_reward + self.discount * max(self.Q[(state_rep_new, 0)], self.Q[(state_rep_new, 1)])\
                 - self.Q[(state_rep_old, int(self.last_action))]
            self.Q[(state_rep_old, int(self.last_action))] += self.learning_rate(self.t) * td


        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    dict_lengths = [0]
    scores = []
    max_scores = []
    for ii in range(iters):
        print 'NEW GAME %d' % (ii)
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        i = 0
        while swing.game_loop():
            i += 1

        # Save score history.
        hist.append(swing.score)

        k = learner.Q.keys()
        #k.sort()
        #print k
        print 'dict length: %d' % len(learner.Q)
        dict_lengths.append(len(learner.Q))
        print 'score: %d' % swing.score
        scores.append(swing.score)
        print 'max score: %d' % max(scores)
        max_scores.append(max(scores))
        print
        print
        # Reset the state of the learner.
        learner.reset()
        print
        
    pickle.dump(learner.Q, open('dict.p', 'w'))
    plt.plot(range(iters+1), dict_lengths)
    plt.show()
    plt.plot(range(iters), scores)
    plt.get_current_fig_manager().window.showMaximized()
    plt.savefig('scores.png')
    plt.show()
    plt.plot(range(iters), max_scores)
    plt.show()
    #d = pickle.load(open('C:\\Users\\Madhu\\Dropbox\\School \'15-\'16\\Semester 2\\CS 181\\cs181-practical4\\practical4-code\\dict.p', 'r'))
    #len(d)

    # demonstrate the learner playing the game
    max_score = 0
    while max_score < 10:
        # Make a new monkey object.
        swing = SwingyMonkey(tick_length=50,
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        i = 0
        while swing.game_loop():
            i += 1
        max_score = swing.score
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 300, 0)

	# Save history. 
	np.save('hist',np.array(hist))


