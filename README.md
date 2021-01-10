# Machine_Learning_Quantum_State_Tomography
Machine learning quantum state tomography has been a heuristic method for quantum state tomography. 

This repository contains the pytorch implementation for using generative models from unsupervised learning (RNN or attention-based RNN) to reconstructing quantum states, which is based on Juan Carrasquilla, Giacomo Torlai, Roger G. Melko & Leandro Aolita's interesting [paper](https://www.nature.com/articles/s42256-019-0028-1).

My implementation is largely inspired by Juan's official [tensorflow implementation](https://github.com/carrasqu/POVM_GENMODEL).

# Running experiments

`main.py` is the code for training the network. All adjustable arguments are stored in `args.py`. They can be displayed via `python main.py --help`

```
usage: main.py [-h] [--N N] [--Ns NS] [--state STATE] [--charset_length CHARSET_LENGTH]
               [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--dtype {float32,float64}]
               [--batch_size BATCH_SIZE] [--lr LR] [--weight_decay WEIGHT_DECAY] [--epoch EPOCH]
               [--clip_grad CLIP_GRAD] [--no_stdout] [--print_step PRINT_STEP] [--save_epoch SAVE_EPOCH]
               [--keep_epoch KEEP_EPOCH] [--cuda CUDA] [--out_infix OUT_INFIX] [-o OUT_DIR]

optional arguments:
  -h, --help            show this help message and exit

network parameters:
  --N N                 number of qubits
  --Ns NS               number of experiments
  --state STATE         quantum state
  --charset_length CHARSET_LENGTH
                        number of char set length
  --hidden_size HIDDEN_SIZE
                        hidden state size in RNN
  --num_layers NUM_LAYERS
                        number of RNN units(depth) in one step
  --dtype {float32,float64}
                        dtype

optimizer parameters:
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               learning rate
  --weight_decay WEIGHT_DECAY
                        weight decay
  --epoch EPOCH         number of epoches
  --clip_grad CLIP_GRAD
                        global norm to clip gradients, 0 for disabled

system parameters:
  --no_stdout           do not print log to stdout, for better performance
  --print_step PRINT_STEP
                        number of batches to print log, 0 for disabled
  --save_epoch SAVE_EPOCH
                        number of epochs to save network weights, 0 for disabled
  --keep_epoch KEEP_EPOCH
                        number of epochs to keep saved network weights, 0 for disabled
  --cuda CUDA           IDs of GPUs to use, empty for disabled
  --out_infix OUT_INFIX
                        infix in output filename to distinguish repeated runs
  -o OUT_DIR, --out_dir OUT_DIR
                        directory for output, empty for disabled
```

During training, the log file and the network weights will be saved in `out_dir`.

# Maintenance

I enjoyed reading those works, and I will add the attention-based RNN implementation later, which has been shown to perform better than GRU-RNN.

