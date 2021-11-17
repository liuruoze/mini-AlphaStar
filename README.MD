# mini-AlphaStar

## Introduction

We release mini-AlphaStar (mAS), which is a mini-scale version of the AlphaStar (AS) program. AS is the intelligent AI proposed by DeepMind to play StarCraft II (SC2). 

"mini-scale" means that we make the original AS's hyper-parameters adjustable so that mAS can be training and running on a small scale (e.g., in a single common commercial machine).

When designing the mAS, we referred to the "Occam's Razor Principle", simple is good. Therefore, we build mAS from scratch. Unless the function has a great impact on speed and performance, we shall omit it. 

At the same time, we also try not to use too many dependency packages, so that mAS should only depend on PyTorch (at most specify the version). In this way, we greatly simplify the learning cost of mAS and make the architecture and code of mAS relatively easy to read. 

In addition, we have added a lot of comments in the code to ensure that some key designs can be conveyed to the readers.

The [Chinese](doc/README_CHS.MD) shows a simple readme in Chinese.

## Update

This is the "v_1.04" version, and the updates are as follows:

* Add mask for supervised learning loss calculation;
* Add camera and non-camera accuracy for evaluation;
* Refactor most code for multi-GPU training;
* Add support for multi-GPU supervised training;

## Location

The codes are stored in two places. 

Location | Usage
------------ | -------------
[Github](https://github.com/liuruoze/mini-AlphaStar) | for global users
[Gitee](https://gitee.com/liuruoze/mini-AlphaStar) | for users who are not fast to access Github easily in China

## Contents

The table below shows the corresponding packages in the project.

Packages | Content
------------ | -------------
alphastarmini.core.arch | deep neural architecture
alphastarmini.core.sl | supervised learning
alphastarmini.core.rl | reinforcement learning
alphastarmini.core.ma | multi-agent league traning
alphastarmini.lib | lib functions
alphastarmini.third | third party functions

## Requirements

PyTorch >= 1.5, others please see requirements.txt.

## Install

The [SCRIPT Guide](scripts/Setup_cmd.MD) gives some commands to install PyTorch by conda (this will automatically install CUDA and cudnn, which is convenient).

E.g., like:
```
conda create -n torch_1_5 python=3.7 pytorch=1.5 -c pytorch
```
to install PyTorch 1.5 (with accompanied CUDA and cudnn).

After install PyTorch, use the following command to active the conda environment:
```
source activate torch_1_5 
```

In Windows, this command may be:
```
conda activate torch_1_5 
```

Then you can install other python packages by pip, e.g., the command in the below line:
```
pip install -r requirements.txt
```

## Usage

After all requirements are done, run the below python file to run the program:
```
python run.py
```
to start the program (you may use comments and uncomments to select the training process you want).

The [USAGE Guide](doc/USAGE.MD) may provide some answers to problems and questions.

## Replays processing

In supervised learning, you need to first download many SC2 replays.

The [REPLAY Guide](doc/REPLAY.MD) shows a guide to download these SC2 replays.

The [ZHIHU Guide](https://zhuanlan.zhihu.com/p/410523216) provides Chinese users who are not convenient to use Battle.net (outside China) a guide to download replays.

After downloading replays, you should make the replays in "./data/Replays/filtered_replays_1".

After downloading replays, you should make the replays in "./data/Replays/filtered_replays_1".

Then use transform_replay_data.py to transform these replays to pickles or tensors (the output type can be changed in the code of that file).

Pickles are Python pickle data that are not preprocessed. Tensors are PyTorch tensors that have been preprocessed.

## Multi-GPU training

This version starts to support multi-GPU supervised learning training for mAS, which can greatly improve the training speed. The way to use multi-GPU training is very straightforward, as follows:
```
python run_multi-gpu.py
```

(Currently, multi-GPU training for RL is still not supported)

Notice:

Multi-GPU training may have unstable factors (this is caused by the reason that PyTorch's support is not as stable as imagined). If you find that multi-GPU training has training instability and runtime errors, please switch back to single-GPU training.

Currently, we support four types of supervised training. They all reside in the package of "alphastarmini.core.sl".

File | Content
------------ | -------------
sl_train_by_pickle.py | pickle training (pickles are data not preprocessed). Slow, but cost small disk space.
sl_train_by_tensor.py | tensor training  (tensors are data preprocessed). Fast, but cost huge disk space.
sl_multi_gpu_by_pickle.py | multi-GPU, pickle training. It has a requirement need for large shared memory.
sl_multi_gpu_by_tensor.py | multi-GPU, tensor training. It needs both for large memory and shared memory.

You can choose the suitable way you need to do training.

You can use the load_pickle.py to transform the generated pickles (in "./data/replay_data") to tensors (in "./data/replay_data_tensor").

## Results

Here are some illustration figures of SL training process below:

![SL training process](doc/SL_traing.png)

We can see the losses (one main loss and six argument losses of which the main loss consists) fall down quickly.

## History

The [HISTORY](doc/HISTORY.MD) is the history introduction of the previous versions of mAS.


## Citing

If you find our repository useful, please cite this project:
```
@misc{liu2021mAS,
  author = {Ruo{-}Ze Liu and Wenhai Wang and Yang Yu and Tong Lu},
  title = {mini-AlphaStar},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuruoze/mini-AlphaStar}},
}
```

## Report

The [An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890) is a technical report that introduces mAS.


## Rethinking

The [Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452) gives some rethinking of AlphaStar.


## Paper

We will give a paper that may be available in the future presenting the experiments and evaluations on using mAS. 