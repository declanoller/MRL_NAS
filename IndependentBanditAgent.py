import matplotlib.pyplot as plt
import numpy as np



class IndependentBanditAgent:


    def __init__(self, **kwargs):

        self.p_arm1 = np.random.random()
        self.p_arm2 = np.random.random()

        self.p = [self.p_arm1, self.p_arm2]

        self.N_actions = 2
        self.N_state_terms = len(self.getStateVec())

        self.trial_counter = None

        self.N_trials = kwargs.get('N_trials', 100)




    def __str__(self):
        return('p1 = {:.2f}, p2 = {:.2f}'.format(*self.p))

    ###################### Required agent functions


    def getStateVec(self):
        return([0])


    def reward(self):
        pass


    def initEpisode(self):
        self.trial_counter = 0


    def iterate(self, action):

        if np.random.rand() <= self.p[action]:
            r = 1.0
        else:
            r = 0.0

        self.trial_counter += 1

        if self.trial_counter >= self.N_trials:
            done = True
        else:
            done = False

        return(r, self.getStateVec(), done)


    def resetStateValues(self):
        pass




#
