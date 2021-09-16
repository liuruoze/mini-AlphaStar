# mini-AlphaStar

## Introduction

We release mini-AlphaStar (mAS), which is a mini-scale version of the AlphaStar (AS) program. AS is the intelligent AI proposed by DeepMind to play StarCraft II (SC2).

"mini-scale" means that we make the original AS's hyperparameters adjustable so that mAS can run on a small scale.

The [Chinese](doc/README_CHS.MD) shows a simple readme in Chinese.

## Update

This is the "v_1.03" version, and the updates are as follows:

* Add guides for "how run RL?" in USAGE.MD;
* Fix some bugs and a bug in SL training;
* Add USAGE.MD;
* Fix a bug by split the map_channels and scatter_channels;
* Fix the right version to see replays of AlphaStar Final Terran;
* Change from z_2 = F.relu(self.fc_2(z_1)) to z_2 = self.fc_2(F.relu(z_1)) in selected_units_head;

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
alphastarmini.core.sl | surpervised learning
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


## Replays

In supervised learning, you need to first download many SC2 replays.

The [REPLAY Guide](doc/REPLAY.MD) shows a guide to download these SC2 replays.

The [ZHIHU Guide](https://zhuanlan.zhihu.com/p/410523216) provides Chinese users who are not convenient to use Battle.net (outside China) a guide to download replays.

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