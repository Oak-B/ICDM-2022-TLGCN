# ICDM-2022-TLGCN
This is the PyTorch implementation for the paper entitled "A Two-Stream Light Graph Convolution Network-based Latent Factor Model for Accurate Cloud Service QoS Estimation", which has been acceppted by ICDM2022.

## Enviroment Requirement
We implement all the experiments in Python 3.7, except that the compressed sparse matrix parallel program is written with CUDA C and compiled with CUDA 11.1. All empirical tests are uniformly deployed on a server with a 2.4-GHz Intel Xeon 4214R CPU, four NVIDIA RTX 3090 GPUs, and 128-GB RAM.

`pip install -r requirements.txt`
