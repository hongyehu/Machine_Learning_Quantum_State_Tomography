import os
import time
import traceback
from math import log, sqrt

import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

import utils
from args import args
from utils import my_log
from rnn import RNN
import numpy as np
from ghz_mps import GHZ
import random

torch.backends.cudnn.benchmark = True

def classical_fidelity(model, mps_state, print_prob = True):
    sample = model.generate(2000).to(args.device)
    prob_model = model.log_prob(sample).exp().detach().cpu().numpy()
    prob_true = mps_state.batch_prob(sample.detach().cpu().numpy())
    if print_prob:
        print('prob_model: ', prob_model[:30])
        print('prob_true: ', prob_true[:30])
    return np.mean(np.sqrt(prob_true/prob_model))
def random_data(batch_size, Nqubits, filename):
    data = np.zeros((batch_size, Nqubits)).astype(int)
    SOS = 4*np.ones((batch_size,1))
    with open(filename) as f:
        lines = f.readlines()
        for i in range(batch_size):
            data[i,:] = np.array(list(map(int, random.choice(lines).split(' ')[:Nqubits]))).astype(int)
    return np.concatenate((SOS, data),axis = 1).astype(np.int)
def cfid(model, mps_state, filename):
    r_data = np.array(random_data(10000,args.N, filename)).astype(np.int)
    prob_true = mps_state.batch_prob(r_data[:,:])
    prob_model = model.log_prob(torch.from_numpy(r_data)).exp().detach().numpy()
    print('prob_model: ', prob_model)
    print('prob_true: ', prob_true)
    return np.mean(np.sqrt(prob_model/prob_true))


def main():
    utils.init_out_dir()
    last_epoch = utils.get_last_checkpoint_step()
    if last_epoch >= args.epoch:
        exit()
    if last_epoch >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_epoch))
    else:
        utils.clear_log()

    model = RNN(args.device, Number_qubits = args.N,charset_length = args.charset_length,\
        hidden_size = args.hidden_size, num_layers = args.num_layers)

    model.train(False)
    print('number of qubits: ', model.Number_qubits)
    my_log('Total nparams: {}'.format(utils.get_nparams(model)))
    model.to(args.device)
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    if last_epoch >= 0:
        utils.load_checkpoint(last_epoch, model, optimizer)

    # Quantum state
    ghz = GHZ(Number_qubits=args.N)
    c_fidelity = classical_fidelity(model, ghz)
    # c_fidelity = cfid(model, ghz, './data.txt')
    print(c_fidelity)

if __name__ == '__main__':
    main()








