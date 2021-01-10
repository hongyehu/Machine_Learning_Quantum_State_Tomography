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

torch.backends.cudnn.benchmark = True

def prepare_data(Nqubits, filename):
    data = list([])
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            data.append(list(map(int, line.split(' ')[:Nqubits])))
    data = np.array(data).astype(np.int)
    BOS = 4*np.ones((data.shape[0],1))
    return np.concatenate((BOS, data),axis = 1).astype(np.int)

def main():
    start_time = time.time()

    utils.init_out_dir()
    last_epoch = utils.get_last_checkpoint_step()
    if last_epoch >= args.epoch:
        exit()
    if last_epoch >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_epoch))
    else:
        utils.clear_log()
    utils.print_args()

    model = RNN(args.device, Number_qubits = args.N,charset_length = args.charset_length,\
        hidden_size = args.hidden_size, num_layers = args.num_layers)
    data = prepare_data(args.N, './data.txt')

    model.train(True)
    my_log('Total nparams: {}'.format(utils.get_nparams(model)))
    model.to(args.device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     model = utils.data_parallel_wrap(model)

    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    if last_epoch >= 0:
        utils.load_checkpoint(last_epoch, model, optimizer)


    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))

    my_log('Training...')
    start_time = time.time()
    for epoch_idx in range(last_epoch + 1, args.epoch + 1):
        for batch_idx in range(int(args.Ns/args.batch_size)):
            optimizer.zero_grad()
            # idx = np.random.randint(low=0,high=int(args.Ns-1),size=(args.batch_size,))
            idx = np.arange(args.batch_size)+batch_idx*args.batch_size
            train_data = data[idx]
            loss = -model.log_prob(torch.from_numpy(train_data).to(args.device)).mean()
            loss.backward()
            if args.clip_grad:
                clip_grad_norm_(params, args.clip_grad)
            optimizer.step()
            if batch_idx == 0:
                my_log('epoch_idx {} loss {:.8g} time {:.3f}'.format(
                    epoch_idx, loss.item(), time.time()-start_time
                    ))
        if (args.out_filename and args.save_epoch
                and epoch_idx % args.save_epoch == 0):
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state,
                       '{}_save/{}.state'.format(args.out_filename, epoch_idx))
            if epoch_idx > 0 and (epoch_idx - 1) % args.keep_epoch != 0:
                os.remove('{}_save/{}.state'.format(args.out_filename,
                                                    epoch_idx - 1))
if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()



















