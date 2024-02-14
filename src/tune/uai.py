import os
import sys
import time
sys.path[0] = os.getcwd()
print(sys.path)
import numpy as np
from train_split import DCATTrainer

DATA = 'uai'
GPU = 0

# 1. step:
base_cmd = [
    '--dataset', DATA,
    '--gpu', GPU,
    '--epochs', 2000,
    '--es-patience', 2000,
    '--lr', 0.005,
    '--early-stop',
    '--model', 'attn_s_smooth_bn_ortho',
    '--beta', 0.1,
    '--gamma', 1,
    '--complement', 'lap-sys-norm',
    '--in-drop', 0.3,
    '--attn-drop', 0.3,
    '--num-hidden', 8,
    '--reduced-dim', 128,
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
