import os
import sys
import time
sys.path[0] = os.getcwd()
print(sys.path)
import numpy as np
from train_split import DCATTrainer

DATA = 'pubmed'
GPU = 1

# 1. step:
base_cmd = [
    '--dataset', DATA,
    '--gpu', GPU,
    '--num-out-heads', 8,
    '--epochs', 1000,
    '--es-patience', 1000,
    '--early-stop',
    '--model', 'attn_s_smooth_bn_ortho',
    '--beta', 0.7,
    '--gamma', 1,
    '--complement', 'lap-sym-norm',
    '--in-drop', 0.6,
    '--attn-drop', 0.6,
    '--num-hidden', 8,
]
i = 0
filename = f'results/{DATA}/{DATA}_dcat_{i}.log'
if not os.path.exists(f'results/{DATA}/'):
    os.makedirs(f'results/{DATA}/')
with open(filename, 'w') as f:
    sys.stdout = f
    base_cmd = [str(c) for c in base_cmd]
    args = DCATTrainer.get_args(base_cmd)
    trainer = DCATTrainer(args)
    t0 = time.time()
    for s in range(40, 50):  # with 10 runs
        cmd = base_cmd + ['--seed', str(s)]
        args = DCATTrainer.get_args(cmd)
        trainer.reset_args(args)
        trainer.train()
    print('\n\nTotal time {:.4f}s'.format(time.time() - t0))
