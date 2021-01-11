import numpy as np
# import os
# import itertools as it
# import sys
import torch
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F
import random
import torch.distributions as dist
from ghz_mps import GHZ
from ncon import ncon
from args import args



class RNN(nn.Module):
    def __init__(self, device, Number_qubits = 6, charset_length = 4, hidden_size=100, num_layers=3):
        super(RNN, self).__init__()
        '''
        char set: 0, 1, 2, 3 are pauli set, and BOS will be a trainable vector
        '''
        self.hidden_size = hidden_size
        self.charset_length = charset_length
        self.Number_qubits = Number_qubits
        self.num_layers = num_layers
        self.embedding = nn.Embedding(charset_length, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, charset_length)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.BOS = nn.Parameter(torch.zeros(hidden_size))
        self.init_hidden = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))
        self.device = device
    def forward(self, a, hidden):
        '''
        a:[bs,]
        input 1: L,bs,H_in
        input 2:S,bs,H_out, S is num_layers
        '''
        embedded = self.embedding(a).unsqueeze(0) #[1,bs, hiddensize]
        output, hidden = self.gru(embedded, hidden)
        output = self.logsoftmax(self.out(output[0]))
        return output, hidden
    # def initHidden(self, batch_size):
    #     return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
    def generate(self, bs):
        a = torch.zeros((bs, self.Number_qubits)).type(torch.LongTensor).to(args.device)
        hidden = self.init_hidden.repeat(1,bs,1)
        # BOS input
        beginning = self.BOS.view(1,1,-1)
        beginning = beginning.repeat(1,bs,1)
        output, hidden = self.gru(beginning, hidden)
        output = self.logsoftmax(self.out(output[0]))
        sampled_op = dist.Categorical(output.squeeze(0).exp()).sample()
        a[:,0] = sampled_op 
        for i in range(0, self.Number_qubits-1):
            output, hidden = self.forward(a[:,i], hidden) #output: [1,bs,charset_length]
            sampled_op = dist.Categorical(output.squeeze(0).exp()).sample()
            a[:,i+1] = sampled_op            
        return a
    def log_prob(self, a):
        # a:[bs, Num_qubit] notice we already add SOS in the beginning
        bs = a.shape[0]
        log_prob = a.new_zeros(a.shape[0])
        hidden = self.init_hidden.repeat(1,bs,1)
        # BOS input
        beginning = self.BOS.view(1,1,-1)
        beginning = beginning.repeat(1,bs,1)
        output, hidden = self.gru(beginning, hidden)
        output = self.logsoftmax(self.out(output[0]))
        log_prob_ = torch.gather(output,1,a[:,0].unsqueeze(1))
        log_prob = log_prob + log_prob_.squeeze(1)
        for i in range(0, self.Number_qubits-1):
            output, hidden = self.forward(a[:,i], hidden)
            log_prob_ = torch.gather(output,1,a[:,i+1].unsqueeze(1))
            log_prob = log_prob + log_prob_.squeeze(1)
        return log_prob




        