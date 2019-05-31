import numpy as np
from copy import deepcopy
from HyperEPANN import HyperEPANN
import FileSystemTools as fst
import matplotlib.pyplot as plt
import torch

'''

Last time, I had the EPANN class create an Agent object, but that doesn't really
make much sense -- the agent is really using the EPANN, not the other way around.
So here, EvoAgent creates a HyperEPANN object, which it uses.


So last time I had an GymAgent class and it made an env. for each one. I think that
was causing some real problems, so this time I think the smart thing to do is either
pass an env object that every EvoAgent will share, or, if nothing is passed, it will
create its own.


agent class (GymAgent or something like Walker_1D) needs members and methods:

-getStateVec()
-initEpisode()
-iterate(action) (returns (reward, state, done))
-drawState()

-state labels (list of strings corresponding to each state)
-action labels (same but with actions)
-action space type ('discrete' or 'continuous')
-N_state_terms
-N_actions



'''


class EvoAgent:

    def __init__(self, **kwargs):

        # Agent stuff
        self.evo_agent_class = kwargs.get('evo_agent_class', None)
        assert self.evo_agent_class is not None, 'Need to provide an agent class! exiting'

        self.agent = self.evo_agent_class(**kwargs)

        self.verbose = kwargs.get('verbose', False)
        self.action_space_type = self.agent.action_space_type
        self.render_type = self.agent.render_type

        # So these are for the agent that EvoAgent is training.
        self.N_inputs = self.agent.N_state_terms
        self.N_outputs = self.agent.N_actions

        # This is for EvoAgent itself: what NN building actions it can do.
        # It needs to have an action term for each parent atom, each child
        # atom, and the three actions it can do to the NN with those terms.
        self.N_max_atoms = kwargs.get('N_max_atoms', 15)
        self.N_choices = 4
        self.N_actions = 2*self.N_max_atoms + self.N_choices

        # HyperEPANN stuff
        self.NN = HyperEPANN(N_inputs=self.N_inputs, N_outputs=self.N_outputs, **kwargs)

        self.NN_output_type = kwargs.get('NN_output_type', 'argmax')



    def __str__(self):

        return('N_atoms: {}, N_weights: {}'.format(len(self.NN.atom_list), len(self.NN.weights_list)))



############################# For interacting with this agent

    def iterate(self, parent_index, child_index, choice_index):
        '''
        So, action input is:
        [one hot array of parent atom]
        [one hot array of child atom]
        [one hot array corresponding to add atom, add weight, remove weight]

        Then, we do the action (if it's legal), changing the NN, train the
        new NN, then eval it to get the avg score.

        '''

        '''N_mutate_actions = 3
        correct_len = (2*self.N_max_atoms + N_mutate_actions)
        assert len(action)==correct_len, \
            f'Wrong size of action input: should be {correct_len}, is {len(action)}'

        OHE_parent_atom = action[:self.N_max_atoms]
        OHE_child_atom = action[self.N_max_atoms:-N_mutate_actions]
        OHE_action_choice = action[-N_mutate_actions:]

        parent_index = np.argmax(OHE_parent_atom)
        child_index = np.argmax(OHE_child_atom)
        choice_index = np.argmax(OHE_action_choice)'''

        legal_check_fns = [
            self.NN.is_legal_atom_add,
            self.NN.is_legal_weight_add,
            self.NN.is_legal_weight_remove,
            self.NN.dummy_legal_fn
        ]

        weight_parchild_tuple = (parent_index, 0, child_index, 0)

        action_functions = [
            self.NN.addAtomInBetween,
            self.NN.addConnectingWeight,
            self.NN.removeConnectingWeight,
            self.NN.dummy_fn
        ]

        function_kwargs = [
            {'atom_type' : 'Node'},
            {'val' : None, 'std' : 0.1},
            {},
            {}
        ]

        legal_fn = legal_check_fns[choice_index]
        action_fn = action_functions[choice_index]
        fn_kwargs = function_kwargs[choice_index]

        if legal_fn(parent_index, child_index):
            if not (choice_index==0 and len(self.NN.atom_list)>=self.N_max_atoms):
                action_fn(weight_parchild_tuple, **fn_kwargs)


        ret = self.runEpisode(50)
        return(ret, 0, 0)


############################## For interfacing with NN


    def forwardPass(self, state_vec):
        #print('state vec', state_vec)
        state_tensor = torch.tensor(state_vec, dtype=torch.float, requires_grad=False)
        output_vec = self.NN.forwardPass(state_tensor)

        if self.NN_output_type == 'argmax':
            a = self.greedyOutput(output_vec)
        else:
            a = output_vec

        return(a)


    def greedyOutput(self, vec):
        return(np.argmax(vec))


    def mutate(self, std=0.1):

        self.NN.mutate(std=std)


    def getNAtoms(self):
        return(len(self.NN.atom_list))


    def getNConnections(self):
        return(len(self.NN.weights_list))


    def plotNetwork(self, **kwargs):
        self.NN.plotNetwork(**kwargs)


    def saveNetworkAsAtom(self, **kwargs):
        self.NN.saveNetworkAsAtom(**kwargs)


    def saveNetworkToFile(self, **kwargs):
        self.NN.saveNetworkToFile(**kwargs)


    def loadNetworkFromFile(self, **kwargs):
        self.NN.loadNetworkFromFile(**kwargs)


    def clone(self):
        clone = deepcopy(self)
        return(clone)


########################### For interacting with the agent class


    def iterate_agent(self, action):

        r, s, done, correct_answer = self.agent.iterate(action)
        #print(correct_answer)
        return(r, s, done, correct_answer)


    def initEpisode(self):
        self.NN.resetOptim()
        self.agent.initEpisode()


################################ For interfacing with gym env and playing


    def setMaxEpisodeSteps(self, N_steps):
        self.agent.setMaxEpisodeSteps(N_steps)



    def runEpisode(self, N_train_steps, **kwargs):


        R_tot = 0
        Rs = []

        show_episode = kwargs.get('show_episode', False)
        record_episode = kwargs.get('record_episode', False)

        N_eval_steps = kwargs.get('N_eval_steps', max(1, N_train_steps//10))

        if show_episode:
            self.createFig()

        if record_episode:
            self.agent.setMonitorOn(show_run=show_episode)

        self.initEpisode()
        train_curve = []

        for i in range(N_train_steps):
            self.NN.clearAllAtoms()

            if i%max(int(N_train_steps/10), 1)==0:
                self.print('R_tot = {:.3f}'.format(R_tot/(i+1)))


            s = self.agent.getStateVec()
            a = self.forwardPass(s)
            self.print('s = {}, a = {}'.format(s, a))

            (r, s, done, correct_answer) = self.iterate_agent(a.detach().numpy())

            l = self.NN.backProp(a, torch.tensor(correct_answer, dtype=torch.float32))
            train_curve.append(l)

            R_tot += r
            Rs.append(R_tot)

            if done:
                #return(R_tot)
                break

            if show_episode or record_episode:
                self.drawState()

        R_tot = 0
        # one after training
        self.initEpisode()
        for i in range(N_eval_steps):
            self.NN.clearAllAtoms()

            s = self.agent.getStateVec()
            a = self.forwardPass(s)

            (r, s, done, correct_answer) = self.iterate_agent(a.detach().numpy())

            R_tot += r
            Rs.append(R_tot)

            if done:
                #return(R_tot)
                break

        #self.agent.closeEnv()
        ret = {}
        ret['r_avg'] = R_tot/N_eval_steps
        ret['train_curve'] = train_curve
        #return(ret)
        return(R_tot/N_eval_steps)




    def drawState(self):

        if self.render_type == 'gym':
            self.agent.drawState()

        if self.render_type == 'matplotlib':
            self.agent.drawState(self.ax)
            self.fig.canvas.draw()






############################# Misc/debugging stuff


    def print(self, str):

        if self.verbose:
            print(str)



    def createFig(self):

        if self.render_type == 'matplotlib':
            self.fig, self.ax = plt.subplots(1, 1, figsize=(4,4))
            plt.show(block=False)



#
