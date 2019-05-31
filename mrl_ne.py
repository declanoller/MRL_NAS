from MetaRL import MetaRL, multi_run
from IndependentBanditAgent import IndependentBanditAgent
from DependentBanditAgent import DependentBanditAgent
from agent_classes import *
from EvoAgent import EvoAgent




mrl = MetaRL(

agent_class = EvoAgent,
evo_agent_class = LogicAgentNand.LogicAgentNand,
N_max_atoms = 10,


gamma = 0.8,
GAE = 0.8,
beta_V_loss = 0.25,
beta_entropy = 0.05,
entropy_method = 'const',
optim = 'Adam',
LR = 10**-3,

clip_grad = True,
clip_amount = 1.0)

mrl.train(N_eps=50, N_steps=500, evo_agent_class = LogicAgentNand.LogicAgentNand, NN_output_type='logic', N_max_atoms = 10)
mrl.save_dat_pickle()
mrl.plot_R_hist()










#
