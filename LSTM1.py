import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import math, random
import matplotlib.pyplot as plt
import pickle
import os


# Define the model

class LSTM1(nn.Module):
    def __init__(self, N_inputs, hidden_size, N_outputs, N_layers=2):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(N_inputs, hidden_size, N_layers, dropout=0)
        self.out_V = nn.Linear(hidden_size, 1)
        self.out_pi = nn.Linear(hidden_size, N_outputs)


    def step(self, input, hidden=None):
        #input = self.inp(input.view(1, -1)).unsqueeze(1)
        #output, hidden = self.rnn(input, hidden)
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output_V = self.out_V(output.squeeze(1))
        output_pi = torch.softmax(self.out_pi(output.squeeze(1)), dim=1)
        return(output_V, output_pi, hidden)

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = torch.tensor(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden
