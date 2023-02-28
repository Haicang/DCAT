import os
import sys
import time
import numpy as np
sys.path[0] = os.getcwd()
print(sys.path)
from train_split import RGATTrainer


index='0'
DATA = 'wiki'
# 1. step:
base_cmd = [
    '--dataset', DATA,
    '--gpu', 0,
    '--model', 'gat',
    '--in-drop', 0.2,
    '--attn-drop', 0.2,
    '--epochs', 2000,
    '--lr', 0.005,
    '--early-stop',
    '--es-patience', 2000,
    '--reduced-dim', 128,
    '--model', 'attn_s_smooth_bn_ortho',
    '--beta', '0.1',
    '--complement', 'lap-rw-norm',
]
if not os.path.exists('results/'):
    os.mkdir('results/')
filename = 'results/{}_dcat_{}.log'.format(DATA, index)
with open(filename, 'w') as f:
    sys.stdout = f
    base_cmd = [str(c) for c in base_cmd]
    args = RGATTrainer.get_args(base_cmd)
    trainer = RGATTrainer(args)
    t0 = time.time()
    for s in range(40, 50):
        cmd = base_cmd + [
            '--seed', str(s),
        ]
        args = RGATTrainer.get_args(cmd)
        trainer.reset_args(args)
        trainer.train()
    print('\n\nTotal time {:.4f}s'.format(time.time() - t0))
