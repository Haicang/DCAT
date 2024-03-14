# Source code of DCAT (Differentiable Clustering for Graph Attention. TKDE 2024)

This repo is available at GitHub.

This repository contains the source code of the paper "Differentiable Clustering for Graph Attention" published in TKDE 2024. Branch `main` contains the original code for the experiments. The packages and their versions are listed in `requirements.txt`. The reorganized code is in the branch `simple`, which adapts to more recent packages  (`requirements.yml`), has simpler APIs and is easier to use.

You can unzip the dataset in `data` directory. And run the code with the following command:
```
python tune/wiki.py
```
under the `src` directory. We find `CUDA=12.1`, `pytorch=2.1.2` and `dgl=2.0.0.cu121` can reproduce the results in the paper.

You can cite the paper as:
```
@article{zhou2024differentiable,
  title={Differentiable Clustering for Graph Attention},
  author={Zhou, Haicang and He, Tiantian and Ong, Yew-Soon and Cong, Gao and Chen, Quan},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```
