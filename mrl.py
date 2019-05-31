from MetaRL import MetaRL, multi_run
from IndependentBanditAgent import IndependentBanditAgent
from DependentBanditAgent import DependentBanditAgent

multi_run(
    N_runs = 3,

    N_eps = 40000,
    N_steps = 100,

    agent_class = IndependentBanditAgent,
    gamma = 0.8,
    GAE = 0.8,
    beta_V_loss = 0.25,
    beta_entropy = 0.05,
    entropy_method = 'const',
    optim = 'Adam',
    LR = 10**-3,

    clip_grad = True,
    clip_amount = 0.5
)


multi_run(
    N_runs = 3,

    N_eps = 20000,
    N_steps = 100,

    agent_class = IndependentBanditAgent,
    gamma = 0.8,
    GAE = 0.8,
    beta_V_loss = 0.25,
    beta_entropy = 0.05,
    entropy_method = 'const',
    optim = 'RMS',
    LR = 10**-3,

    clip_grad = True,
    clip_amount = 1.0
)


multi_run(
    N_runs = 3,

    N_eps = 20000,
    N_steps = 100,

    agent_class = IndependentBanditAgent,
    gamma = 0.8,
    GAE = 0.8,
    beta_V_loss = 0.25,
    beta_entropy = 0.05,
    entropy_method = 'anneal',
    optim = 'Adam',
    LR = 10**-3,

    clip_grad = True,
    clip_amount = 1.0
)


multi_run(
    N_runs = 3,

    N_eps = 20000,
    N_steps = 100,

    agent_class = IndependentBanditAgent,
    gamma = 0.99,
    GAE = 0.99,
    beta_V_loss = 0.25,
    beta_entropy = 0.05,
    entropy_method = 'const',
    optim = 'Adam',
    LR = 10**-3,

    clip_grad = True,
    clip_amount = 1.0
)




multi_run(
    N_runs = 3,

    N_eps = 20000,
    N_steps = 100,

    agent_class = IndependentBanditAgent,
    gamma = 0.8,
    GAE = 0.8,
    beta_V_loss = 0.05,
    beta_entropy = 0.05,
    entropy_method = 'const',
    optim = 'Adam',
    LR = 10**-3,

    clip_grad = True,
    clip_amount = 1.0
)



multi_run(
    N_runs = 3,

    N_eps = 40000,
    N_steps = 100,

    agent_class = IndependentBanditAgent,
    gamma = 0.8,
    GAE = 0.8,
    beta_V_loss = 0.25,
    beta_entropy = 0.05,
    entropy_method = 'const',
    optim = 'Adam',
    LR = 10**-3,
    hidden_size = 96,

    clip_grad = True,
    clip_amount = 1.0
)










#
