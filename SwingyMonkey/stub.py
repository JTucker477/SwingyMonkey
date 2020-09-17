# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import pygame
from SwingyMonkey import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):

        self.action = None
        self.gamma = None
        self.epsilon = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE, 2))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        '''
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree        
        '''

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)

        gravity = int(state['gravity'])
        if gravity == 1:
            gravity = 0
        else:
            gravity = 1
        return (rel_x, rel_y, gravity)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.

        new_state = self.discretize_state(state)
        x = self.discretize_state(state)[0]
        y = self.discretize_state(state)[1]
        gravity = self.discretize_state(state)[2]


        if self.last_action == None:
            answer = npr.rand() < 0.5
            self.last_action = answer
            self.last_state = new_state
            return self.last_action

        if self.last_action == False:
            self.last_action = 0

        if self.last_action == True:
            self.last_action = 1

        alpha = self.alpha
        gamma = self.gamma

        max_q = np.amax(self.Q[:,x,y, gravity])

        old_x = self.last_state[0]
        old_y = self.last_state[1]
        old_gravity = self.last_state[2]
        self.Q[self.last_action][old_x][old_y][old_gravity] = self.Q[self.last_action][old_x][old_y][old_gravity] + alpha*(self.last_reward + gamma*max_q - self.Q[self.last_action][old_x][old_y][old_gravity])

        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.

        # 3. Choose the next action using an epsilon-greedy policy.
        epsilon = self.epsilon

        if npr.rand() < (1 - epsilon):
            new_action = np.argmax(self.Q[:,x,y,gravity])

        else:
            if npr.rand() > .5:
                new_action = 0
            else:
                new_action = 1



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
    for ii in range(iters):


        # Make a new monkey object.
        swing = SwingyMonkey(sound=True,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':


    alpha_test = [0,.1,.2,.3,.4]
    epsilon_test = [0,.020,.04,.06,.08,.1]
    gamma_test = [.7,.75,.8,.9,.95, 1]

    agent = Learner()
    hist = []


    agent.alpha = .2
    agent.gamma = .95

    agent.epsilon = 0

    run_games(agent, hist, 100, 100)

    hyper = []
    for alpha in alpha_test:
        for gamma in gamma_test:
            # Empty list to save history.
            hist = []

            # Select agent.
            agent = Learner()

            agent.alpha = alpha
            agent.gamma = gamma

            agent.epsilon = 0

            run_games(agent, hist, 100, 100)

            d = {'high score': max(hist), 'alpha': alpha,  'gamma' : gamma }
            hyper.append(d)
    newdict = sorted(hyper, key=lambda k:k['high score'],reverse = True)
    print(newdict)
    # Save history.
    np.save('hist',np.array(hist))


