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



class RNN(nn.Module):
    def __init__(self, device, Number_qubits = 6, charset_length = 5, hidden_size=100, num_layers=3):
        super(RNN, self).__init__()
        '''
        char set: 0, 1, 2, 3 are pauli set, 4 is SOS
        '''
        self.hidden_size = hidden_size
        self.charset_length = charset_length
        self.Number_qubits = Number_qubits
        self.num_layers = num_layers
        self.embedding = nn.Embedding(charset_length, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, charset_length)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
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
    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
    def generate(self, bs):
        a = torch.zeros((bs, self.Number_qubits+1)).type(torch.LongTensor)
        a[:,0] = 4*torch.ones(bs).type(torch.LongTensor)
        hidden = self.initHidden(bs)
        for i in range(0, self.Number_qubits):
            output, hidden = self.forward(a[:,i], hidden) #output: [1,bs,charset_length]
            sampled_op = dist.Categorical(output.squeeze(0).exp()).sample()
            a[:,i+1] = sampled_op            
#             topv, topi = output.squeeze(0).topk(1)
#             a[:,i+1] = topi.squeeze().detach()
        return a
    def log_prob(self, a):
        # a:[bs, Num_qubit+1] notice we already add SOS in the beginning
        log_prob = a.new_zeros(a.shape[0])
        hidden = self.initHidden(a.shape[0])
        for i in range(0, self.Number_qubits):
            output, hidden = self.forward(a[:,i], hidden)
            log_prob_ = torch.gather(output,1,a[:,i+1].unsqueeze(1))
            log_prob = log_prob + log_prob_.squeeze(1)
        return log_prob
    def log_prob_without_teacher(self, a):
        # This is not really realible. but why NLP people use it
        decoder_input = 4*torch.ones(a.shape[0]).type(torch.LongTensor)
        log_prob = a.new_zeros(a.shape[0])
        hidden = self.initHidden(a.shape[0])
        for i in range(0, self.Number_qubits):
            output, hidden = self.forward(decoder_input, hidden) #output: [1,bs,charset_length]
#             topv, topi = output.squeeze(0).topk(1)
            decoder_input = sampled_op = dist.Categorical(output.squeeze(0).exp()).sample().detach()
            log_prob_ = torch.gather(output,1,a[:,i+1].unsqueeze(1))
            log_prob = log_prob + log_prob_.squeeze(1)
        return log_prob



        