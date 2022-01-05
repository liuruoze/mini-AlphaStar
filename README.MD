# mini-AlphaStar

## Introduction

The mini-AlphaStar (mini-AS, or mAS) project is a **mini-scale** version (non-official) of the AlphaStar (AS). AlphaStar is the intelligent AI proposed by DeepMind to play StarCraft II. 

The "mini-scale" means making the original AS's hyper-parameters **adjustable** so that mini-AS can be trained and running on a **small** scale. E.g., we can train this model in a single commercial server machine.

We referred to the "Occam's Razor Principle" when designing the mini-AS": **simple** is sound. Therefore, we build mini-AS from scratch. Unless the function significantly impacts speed and performance, we shall omit it. 

Meanwhile, we also try not to use too many dependency packages so that mini-AS should only depend on the PyTorch. In this way, we simplify the learning cost of the mini-AS and make the architecture of mini-AS relatively **easy**. 

The [Chinese](doc/README_CHS.MD) shows a simple readme in Chinese.

Below 4 GIFs are mini-AS' trained performance on Simple64, supervised learning on 50 expert replays.

<p float="left">
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/start.gif" width="360" />
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/middle.gif" width="360" /> 
</p>

**Left:** At the start of the game.     **Right:** In the middle period of the game.

<p float="left">
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/attack1.gif" width="360" />
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/attack2.gif" width="360" />
</p>

**Left:** The agent's 1st attack.      **Right:** The agent's 2nd Attack.

## Update

This release is the "v_1.06" version. In this version, we win the built-in AI for the first time, increase the win rate to a higher value, and provide improved RL training by being more stable, faster, and more concise. Here are the details:
 
* First success in selecting Probes to build Pylons in the correct positions;
* Increase the selection accuracy in SL and initial status of RL;
* Improve the win rate against the built-in AI;
* Increase the win rate against built-in AI to 0.8 and the killed points to 5900;
* Add result videos;
* Improve baseline (alias for state value estimation in RL) in accuracy and speed;
* Improve the reproduce ability of the RL trained results (use random seed and single thread);
* Greatly re-factor the code writing of the RL loss, add the entity mask;
* Fix the big loss problem in the RL loss calculation by the outlier_remove function;
* Reduce the code lines of rl_loss by 58.2%;
* Improve log_prob, KL, entropy coding;
* Reduce the code lines of rl_algo by 31.4%;

## Hints

**Warning**: SC2 is extremely difficult, and AlphaStar is also very complex. Even our project is a mini-AlphaStar, it has almost the similar technologies as AlphaStar, and the training resource also costs very high. We can hardly train Mini-AS on a laptop. The recommended way is to use a commercial server with a GPU card and enough large memory and disk space. For someone interested in this project for the first time, we recommend you collect (star) this project and devolve deeply into researching it when you have enough free time and training resources.

## Location

We store the codes and show videos in two places. 

Codes location | Result video location | Usage 
------------ | ------------- | -------------
[Github](https://github.com/liuruoze/mini-AlphaStar) | [Youtube](https://youtu.be/mTtA0vdAULw) | for global users 
[Gitee](https://gitee.com/liuruoze/mini-AlphaStar) | [Bilibili](https://www.bilibili.com/video/BV1Hm4y197Jm/) | for users in China 

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

E.g., like (to install PyTorch 1.5 with accompanied CUDA and cudnn):
```
conda create -n th_1_5 python=3.7 pytorch=1.5 -c pytorch
```

Next, activate the conda environment, like:
```
conda activate th_1_5
```

Then you can install other python packages by pip, e.g., the command in the below line:
```
pip install -r requirements.txt
```

## Usage

After you have done all requirements, run the below python file to run the program:
```
python run.py
```
You may use comments and uncomments in "run.py" to select the training process you want.

The [USAGE Guide](doc/USAGE.MD) provides answers to some problems and questions.

## Replays processing

In supervised learning, you first need to download SC2 replays.

The [REPLAY Guide](doc/REPLAY.MD) shows a guide to download these SC2 replays.

The [ZHIHU Guide](https://zhuanlan.zhihu.com/p/410523216) provides Chinese users who are not convenient to use Battle.net (outside China) a guide to download replays.

After downloading replays, you should move the replays to "./data/Replays/filtered_replays_1" (you can change the name in `transform_replay_data.py`).

Then use `transform_replay_data.py` to transform these replays to pickles or tensors (you can change the output type in the code of that file).

## Multi-GPU training

From the v_1.05 version, we start to support multi-GPU supervised learning training for mini-AS, improving the training speed. The way to use multi-GPU training is straightforward, as follows:
```
python run_multi-gpu.py
```

Currently, multi-GPU training for RL is still not supported.

Multi-GPU training has some unstable factors (caused because of PyTorch). If you find your multi-GPU training has training instability errors, please switch to the single-GPU training.

We currently support four types of supervised training, which all reside in the "alphastarmini.core.sl" package.

File | Content
------------ | -------------
`sl_train_by_pickle.py` | pickle (data not preprocessed) training: Slow, but need small disk space.
`sl_train_by_tensor.py` | tensor (data preprocessed) training: Fast, but cost colossal disk space.
`sl_multi_gpu_by_pickle.py` | multi-GPU, pickle training: It has a requirement need for large shared memory.
`sl_multi_gpu_by_tensor.py` | multi-GPU, tensor training: It needs both large memory and large shared memory.

You can use the `load_pickle.py` to transform the generated pickles (in "./data/replay_data") to tensors (in "./data/replay_data_tensor").

**Note**: in v_1.06, we still recommend using single-GPU training. We provide the new training ways in the single-GPU type. This is due to multi-GPU training cost so much memory

## Results

Here are some illustration figures of the SL training process below:

![SL training process](doc/SL_traing.png)

We can see the loss (one primary loss and six argument losses) fall quickly.

## History

The [HISTORY](doc/HISTORY.MD) is the historical introduction of the previous versions of mini-AS.


## Citing

If you find our repository useful, please cite our project:
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

The [An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890) is a technical report introducing the mini-AS (not full version).


## Rethinking

The [Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452) is our thinking of the advantages and disadvantages of AlphaStar.


## Paper

We will give a paper (which is now under peer-review) that may be available in the future, presenting detailed experiments and evaluations using the mini-AS. 