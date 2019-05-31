from LSTM_multi_head import LSTM_multi_head
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import decorators
import os
from math import exp, log
import FileSystemTools as fst
import pickle
import json

class MetaRL:


	def __init__(self, **kwargs):

		self.agent_class = kwargs.get('agent_class',None)
		self.agent_class_name = self.agent_class.__name__
		temp_agent = self.agent_class(**kwargs)

		self.N_state_terms = 0
		if hasattr(temp_agent, 'N_state_terms'):
			self.N_state_terms = temp_agent.N_state_terms

		self.N_choices = temp_agent.N_choices
		self.N_max_atoms = kwargs.get('N_max_atoms', 15)
		self.N_actions = 2*self.N_max_atoms + self.N_choices

		self.beta_entropy_min = 0.03
		self.beta_entropy_max = 1.0

		print('\nMetaRL object set up.\n\n')

		self.gamma = kwargs.get('gamma', 0.8)
		self.beta_GAE = kwargs.get('beta_GAE', self.gamma)
		self.entropy_method = kwargs.get('entropy_method', 'const')
		self.beta_entropy = kwargs.get('beta_entropy', 0.05)
		self.beta_V_loss = kwargs.get('beta_V_loss', 0.25)
		self.optim = kwargs.get('optim', 'Adam')
		self.LR = kwargs.get('LR', 10**-3)
		self.hidden_size = kwargs.get('hidden_size', 48)

		self.fname_base = 'g={:.2f}_bH={:.2f}_Hmethod={}_bV={:.2f}_opt={}_LR={}_hid={}__GAE={}__{}'.format(
																				self.gamma,
																				self.beta_entropy,
																				self.entropy_method,
																				self.beta_V_loss,
																				self.optim,
																				self.LR,
																				self.hidden_size,
																				self.beta_GAE,
																				fst.getDateString()
		)
		self.setup_NN()
		self.clip_grad = kwargs.get('clip_grad', False)
		self.clip_amount = kwargs.get('clip_amount', 1.0)

		self.dir = kwargs.get('dir', os.path.join(os.path.dirname(__file__), 'runs'))

		if kwargs.get('save_params', False):
			self.N_eps = kwargs.get('N_eps', None)
			self.N_steps = kwargs.get('N_steps', None)
			self.save_params_json()


	def save_params_json(self):

		params_dict = {
			'agent_class_name' : self.agent_class_name,
			'N_state_terms' : self.N_state_terms,
			'N_actions' : self.N_actions,
			'beta_entropy_min' : self.beta_entropy_min,
			'beta_entropy_max' : self.beta_entropy_max,
			'gamma' : self.gamma,
			'beta_GAE' : self.beta_GAE,
			'entropy_method' : self.entropy_method,
			'beta_entropy' : self.beta_entropy,
			'beta_V_loss' : self.beta_V_loss,
			'optim' : self.optim,
			'LR' : self.LR,
			'hidden_size' : self.hidden_size,
			'clip_grad' : self.clip_grad,
			'clip_amount' : self.clip_amount,
			'N_eps' : self.N_eps,
			'N_steps' : self.N_steps,
		}

		fname = os.path.join(self.dir, 'params.json')

		with open(fname, 'w+') as f:
			json.dump(params_dict, f, indent=4)


	def setup_NN(self):

		N_inputs = self.N_actions + 2 # +1 for reward, +1 for time step
		N_outputs = self.N_actions
		N_layers = 1
		self.NN = LSTM_multi_head(N_inputs, self.hidden_size, self.N_max_atoms, self.N_choices, N_layers)

		for k,v in self.NN.state_dict().items():
			if 'bias' in str(k):
				v.data.fill_(0)

		#self.NN.out_pi.weight.data = self.norm_col_init(self.NN.out_pi.weight.data, 0.01)
		self.NN.out_pi_parent.weight.data = self.norm_col_init(self.NN.out_pi_parent.weight.data, 0.01)
		self.NN.out_pi_child.weight.data = self.norm_col_init(self.NN.out_pi_child.weight.data, 0.01)
		self.NN.out_pi_choice.weight.data = self.norm_col_init(self.NN.out_pi_choice.weight.data, 0.01)
		self.NN.out_V.weight.data = self.norm_col_init(self.NN.out_V.weight.data, 1.0)

		if self.optim == 'Adam':
			self.optimizer = optim.Adam(self.NN.parameters(), lr=self.LR)
		elif self.optim == 'RMS':
			self.optimizer = optim.RMSprop(self.NN.parameters(), lr=self.LR)
		else:
			self.optimizer = optim.SGD(self.NN.parameters(), lr=self.LR)


	def norm_col_init(self, weights, std=1.0):
		x = torch.randn(weights.size())
		x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
		return (x)


	@decorators.timer
	def train(self, N_eps, N_steps, **kwargs):

		print('\nBeginning training...\n')

		self.R_hist = []
		self.a_hist = []
		self.H_coeff_hist = []
		self.policy_loss = []
		self.V_loss = []
		self.entropy_loss = []

		discount_gamma = torch.tensor(
					np.array([[self.gamma**(c-r) if c>=r else 0 for c in range(N_steps)] for r in range(N_steps)])
					, dtype=torch.float)

		discount_beta_GAE = torch.tensor(
					np.array([[self.beta_GAE**(c-r) if c>=r else 0 for c in range(N_steps)] for r in range(N_steps)])
					, dtype=torch.float)

		for ep in range(N_eps):

			# Create a new agent, so it has diff probs for the two arms
			agent = self.agent_class(**kwargs)

			if self.entropy_method == 'const':
				beta_entropy = self.beta_entropy
			else:
				beta_entropy = self.entropy_coeff(ep, N_eps)

			ret = self.episode(agent, N_steps)

			outputs_V = ret['outputs_V']
			outputs_pi_parent = ret['outputs_pi_parent']
			entropy_parent = ret['entropy_parent']
			outputs_pi_child = ret['outputs_pi_child']
			entropy_child = ret['entropy_child']
			outputs_pi_choice = ret['outputs_pi_choice']
			entropy_choice = ret['entropy_choice']
			R = ret['R']

			agent.NN.plotNetwork(show_plot=False, save_plot=True, fname=f'misc_plots/NN_ep{ep}.png')
			plt.clf()
			plt.plot(R.detach().squeeze().numpy())
			plt.title(f'episode {ep}')
			plt.xlabel('iteration')
			plt.ylabel('R')
			plt.savefig(f'misc_plots/R_ep{ep}.png')

			R_tot = R.sum().item()
			self.a_hist.append(ret['a'])
			self.H_coeff_hist.append(beta_entropy)
			#normed_R = (R_tot/N_steps)/(0.5*(agent.p_arm1 + agent.p_arm2))
			normed_R = 1.0
			self.R_hist.append(normed_R)

			r_accum = torch.mm(discount_gamma, R.squeeze(dim=1))

			V_target = torch.cat((outputs_V.squeeze()[1:], torch.tensor([0.0]))).unsqueeze(dim=1)

			adv = R.squeeze(dim=1) + self.gamma*V_target - outputs_V.squeeze(dim=1)
			adv = torch.mm(discount_beta_GAE, adv).squeeze()

			V_loss_tot = self.beta_V_loss*(r_accum.squeeze() - outputs_V.squeeze()).pow(2).sum()

			J_tot_parent = -(torch.log(outputs_pi_parent.squeeze())*adv).sum()
			entropy_tot_parent = beta_entropy*entropy_parent.squeeze().sum()
			J_tot_child = -(torch.log(outputs_pi_child.squeeze())*adv).sum()
			entropy_tot_child = beta_entropy*entropy_child.squeeze().sum()
			J_tot_choice = -(torch.log(outputs_pi_choice.squeeze())*adv).sum()
			entropy_tot_choice = beta_entropy*entropy_choice.squeeze().sum()

			loss_tot = V_loss_tot +\
						J_tot_parent + entropy_tot_parent +\
						J_tot_child + entropy_tot_child +\
						J_tot_choice + entropy_tot_choice
			self.optimizer.zero_grad()
			loss_tot.backward()

			if self.clip_grad:
				nn.utils.clip_grad_norm_(self.NN.parameters(), self.clip_amount)

			self.optimizer.step()

			#self.policy_loss.append(J_tot.item())
			#self.V_loss.append(V_loss_tot.item())
			#self.entropy_loss.append(entropy_tot.item())

			if ep % max(N_eps // 100, 1) == 0:
				print('\n\n\nepisode {}/{}'.format(ep, N_eps))
				print('Cur agent: ', agent)
				print('Final avg reward = {:.3f}'.format(R_tot/N_steps))
				print('(Avg reward)/(random performance) = {:.3f}'.format(normed_R))


	def episode(self, agent, N_steps):

		agent.initEpisode()
		r = 0
		#a = 0
		a_array = []
		# Create tensors filled with zeros, which will be filled in with values
		# as we get them.
		outputs_V = torch.tensor(torch.zeros(N_steps, 1, 1))
		outputs_pi_parent = torch.tensor(torch.zeros(N_steps, 1, 1))
		outputs_pi_child = torch.tensor(torch.zeros(N_steps, 1, 1))
		outputs_pi_choice = torch.tensor(torch.zeros(N_steps, 1, 1))
		R = torch.tensor(torch.zeros(N_steps, 1, 1))
		entropy_parent = torch.tensor(torch.zeros(N_steps, 1, 1))
		entropy_child = torch.tensor(torch.zeros(N_steps, 1, 1))
		entropy_choice = torch.tensor(torch.zeros(N_steps, 1, 1))
		hidden = (torch.randn(1, 1, self.hidden_size), torch.randn(1, 1, self.hidden_size))

		a_parent = 0
		a_child = 0
		a_choice = 0

		for t in range(N_steps):

			a_one_hot_parent = torch.zeros(self.N_max_atoms, dtype=torch.float)
			a_one_hot_parent[a_parent] = 1
			a_one_hot_child = torch.zeros(self.N_max_atoms, dtype=torch.float)
			a_one_hot_child[a_child] = 1
			a_one_hot_choice = torch.zeros(self.N_choices, dtype=torch.float)
			a_one_hot_choice[a_choice] = 1
			r_tensor = torch.tensor([r], dtype=torch.float)
			t_tensor = torch.tensor([t], dtype=torch.float)

			#NN_input = torch.cat((a_one_hot, r_tensor))
			NN_input = torch.cat((
								a_one_hot_parent,
								a_one_hot_child,
								a_one_hot_choice,
								r_tensor,
								t_tensor))

			out_V, out_pi_parent, out_pi_child, out_pi_choice, hidden = self.NN.step(NN_input, hidden)

			a_next_parent = self.softmax_action(out_pi_parent)
			a_next_child = self.softmax_action(out_pi_child)
			a_next_choice = self.softmax_action(out_pi_choice)

			outputs_V[t] = out_V

			outputs_pi_parent[t] = out_pi_parent.squeeze()[a_next_parent]
			outputs_pi_child[t] = out_pi_child.squeeze()[a_next_child]
			outputs_pi_choice[t] = out_pi_choice.squeeze()[a_next_choice]

			entropy_parent[t] = -sum(out_pi_parent.squeeze()*torch.log(out_pi_parent.squeeze()))
			entropy_child[t] = -sum(out_pi_child.squeeze()*torch.log(out_pi_child.squeeze()))
			entropy_choice[t] = -sum(out_pi_choice.squeeze()*torch.log(out_pi_choice.squeeze()))

			a_parent = a_next_parent
			a_child = a_next_child
			a_choice = a_next_choice

			r, s_next, done = agent.iterate(a_parent, a_child, a_choice)
			R[t] = r



		a_perc = sum(a_array)/N_steps
		return({
		'outputs_V' : outputs_V,
		'outputs_pi_parent' : outputs_pi_parent,
		'entropy_parent' : entropy_parent,
		'outputs_pi_child' : outputs_pi_child,
		'entropy_child' : entropy_child,
		'outputs_pi_choice' : outputs_pi_choice,
		'entropy_choice' : entropy_choice,
		'R' : R,
		'a' : a_perc})


	def softmax_action(self, vec):

		m = Categorical(vec)
		return(m.sample().item())


	def entropy_coeff(self, ep_num, N_eps):

		# exp_decrease should go from 1 to 0 over the course of N_eps
		final_val = 10**-4
		decay = -log(final_val)/N_eps
		speed = 3.0
		exp_decrease = exp(-decay*speed*ep_num)
		return( (self.beta_entropy_max - self.beta_entropy_min)*exp_decrease + self.beta_entropy_min )


	def smooth_data(self, in_dat):

		hist = np.array(in_dat)
		N_avg_pts = min(100, len(hist))

		avg_period = len(hist) // N_avg_pts

		downsampled_x = avg_period*np.array(range(N_avg_pts))
		hist_downsampled_mean = [hist[i*avg_period:(i+1)*avg_period].mean() for i in range(N_avg_pts)]
		return(downsampled_x, hist_downsampled_mean)



	def plot_R_hist(self):

		plt.clf()

		fig, axes = plt.subplots(2, 3, figsize=(12,8))

		ax_R = axes[0][0]
		ax_R_smooth = axes[0][1]
		ax_action = axes[0][2]

		ax_pol_loss = axes[1][0]
		ax_V_loss = axes[1][1]
		ax_entropy_loss = axes[1][2]

		ax_R.set_xlabel('Episode')
		ax_R.set_ylabel('(Avg reward)/(random perf.)')
		ax_R.plot(self.R_hist, label='R_hist', color='dodgerblue')

		ax_R_smooth.set_xlabel('Episode')
		ax_R_smooth.set_ylabel('Smoothed (Avg reward)/(random perf.)')
		ax_R_smooth.plot(*self.smooth_data(self.R_hist), label='R_hist', color='forestgreen', marker='.')

		ax_action.set_xlabel('Episode')
		ax_action.set_ylabel('Percent of actions p2')
		ax_action.plot(self.a_hist, label='a_hist', color='darkorange')

		ax_pol_loss.set_xlabel('Episode')
		ax_pol_loss.set_ylabel('Policy loss')
		ax_pol_loss.plot(self.policy_loss, color='lightcoral')
		ax_pol_loss.plot(*self.smooth_data(self.policy_loss), color='black')

		ax_V_loss.set_xlabel('Episode')
		ax_V_loss.set_ylabel('V loss')
		ax_V_loss.plot(self.V_loss, color='plum')
		ax_V_loss.plot(*self.smooth_data(self.V_loss), color='black')

		ax_entropy_loss.set_xlabel('Episode')
		ax_entropy_loss.set_ylabel('Entropy loss')
		ax_entropy_loss.plot(self.entropy_loss, color='paleturquoise')
		ax_entropy_loss.plot(*self.smooth_data(self.entropy_loss), color='black')
		print(len(self.entropy_loss))

		plt.tight_layout()

		plt.savefig(os.path.join(self.dir, 'plots/{}.png'.format(self.fname_base)))


	def save_dat_pickle(self):

		fname = os.path.join(self.dir, 'data/{}.pkl'.format(self.fname_base))

		with open(fname, 'wb') as f:
			pickle.dump({
			'R_hist' : self.R_hist,
			'H_coeff_hist' : self.H_coeff_hist,
			'a_hist' : self.a_hist
			},
			f)


##################### Multi run

def multi_run(N_runs, **kwargs):

	dir_name = os.path.join('runs', get_fname_base(**kwargs))
	os.mkdir(dir_name)
	print(f'\n\nMade dir {dir_name} for runs...\n\n')
	os.mkdir(os.path.join(dir_name, 'plots'))
	os.mkdir(os.path.join(dir_name, 'data'))

	for i in range(N_runs):

		print(f'\n\nStarting run {i+1}/{N_runs}\n\n')

		if i == 0:
			mrl = MetaRL(**kwargs, dir=dir_name, save_params=True)
		else:
			mrl = MetaRL(**kwargs, dir=dir_name, save_params=False)

		mrl.train(kwargs.get('N_eps', 100), kwargs.get('N_steps', 100))
		mrl.save_dat_pickle()
		mrl.plot_R_hist()






def get_fname_base(**kwargs):

	gamma = kwargs.get('gamma', 0.8)
	beta_GAE = kwargs.get('beta_GAE', gamma)
	entropy_method = kwargs.get('entropy_method', 'const')
	beta_entropy = kwargs.get('beta_entropy', 0.05)
	beta_V_loss = kwargs.get('beta_V_loss', 0.25)
	optim = kwargs.get('optim', 'Adam')
	LR = kwargs.get('LR', 10**-3)
	hidden_size = kwargs.get('hidden_size', 48)

	fname_base = 'g={:.2f}_bH={:.2f}_Hmethod={}_bV={:.2f}_opt={}_LR={}_hid={}__GAE={}__{}'.format(
																			gamma,
																			beta_entropy,
																			entropy_method,
																			beta_V_loss,
																			optim,
																			LR,
																			hidden_size,
																			beta_GAE,
																			fst.getDateString()
	)

	return(fname_base)








#
