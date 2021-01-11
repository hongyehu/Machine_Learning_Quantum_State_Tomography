import argparse
import os
from math import log2

import torch

parser = argparse.ArgumentParser()

############################################
group = parser.add_argument_group('network parameters')
group.add_argument('--N', type=int, default=40, help='number of qubits')
group.add_argument('--Ns', type = int, default = 100000, help = 'number of experiments')
group.add_argument('--state', type=str, default= 'GHZ', help='quantum state')
group.add_argument('--charset_length', type=int, default = 4, help = 'number of char set length')
group.add_argument('--hidden_size', type=int, default = 100, help='hidden state size in RNN')
group.add_argument('--num_layers', type=int, default = 3, help='number of RNN units(depth) in one step')
group.add_argument('--dtype',
                   type=str,
                   default='float64',
                   choices=['float32', 'float64'],
                   help='dtype')
############################################
group = parser.add_argument_group('optimizer parameters')
group.add_argument('--batch_size', type=int, default=1000, help='batch size')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument('--weight_decay',
                   type=float,
                   default=5e-5,
                   help='weight decay')
group.add_argument('--epoch', type=int, default=100000, help='number of epoches')
group.add_argument('--clip_grad',
                   type=float,
                   default=1,
                   help='global norm to clip gradients, 0 for disabled')
############################################
group = parser.add_argument_group('system parameters')
group.add_argument('--no_stdout',
                   action='store_true',
                   help='do not print log to stdout, for better performance')
group.add_argument('--print_step',
                   type=int,
                   default=1,
                   help='number of batches to print log, 0 for disabled')
group.add_argument(
    '--save_epoch',
    type=int,
    default=1,
    help='number of epochs to save network weights, 0 for disabled')
group.add_argument(
    '--keep_epoch',
    type=int,
    default=10,
    help='number of epochs to keep saved network weights, 0 for disabled')
group.add_argument('--cuda',
                   type=str,
                   default='',
                   help='IDs of GPUs to use, empty for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument('-o',
                   '--out_dir',
                   type=str,
                   default='./saved_model',
                   help='directory for output, empty for disabled')

args = parser.parse_args()
############################################
if args.dtype == 'float32':
    torch.set_default_tensor_type(torch.FloatTensor)
elif args.dtype == 'float64':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

if args.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda')
    args.device_count = len(args.cuda.split(','))
else:
    args.device = torch.device('cpu')
    args.device_count = 1


def get_net_name():
    net_name = 'state{}N{}char{}nh{}layers{}'.format(args.state, args.N, args.charset_length,\
      args.hidden_size, args.num_layers)
    return net_name


args.net_name = get_net_name()

if args.out_dir:
    args.out_filename = os.path.join(
        args.out_dir,
        args.net_name,
        'out{out_infix}'.format(**vars(args)),
    )

else:
    args.out_filename = None
    args.plot_filename = None


def str_to_int_list(s, depth):
    if ',' in s:
        out = []
        for x in s.split(','):
            x = int(x)
            out += [x, x]
        return out
    else:
        return [int(s)] * depth
