# mini-AlphaStar


## Update

This is the "v_1.03" version, and the updates are as follows:

* Add guides for "how run RL?" in USAGE.MD;
* Fix some bugs and a bug in SL training;
* Add USAGE.MD;
* Fix a bug by split the map_channels and scatter_channels;
* Fix the right version to see replays of AlphaStar Final Terran;
* Change from z_2 = F.relu(self.fc_2(z_1)) to z_2 = self.fc_2(F.relu(z_1)) in selected_units_head;

## Introduction

We release the mini-AlphaStar (mAS) project, which is a mini source version of the original AlphaStar program. AlphaStar is an intelligent AI proposed by DeepMind to play StarCraft II. StarCraft II is an RTS game developed by Blizzard.

"mini" means that we make the original AlphaStar hyperparameter adjustable so that it can run on a small scale.

The readme for the Chinese version is at [here](doc/README_CHS.MD).

## Contents

The below table shows the corresponding packages in the project.

Packages | Content
------------ | -------------
alphastarmini.core.arch | the alphaStar architecture
alphastarmini.core.sl | surpervised learning
alphastarmini.core.rl | reinforcement learning
alphastarmini.core.ma | multi-agent league traning
alphastarmini.lib | lib functions
alphastarmini.third | third party functions
res | other useful resources

## Requirements

Pytorch >= 1.5, others please see requirements.txt.

## Location

The codes are in these places:

Location | URL
------------ | -------------
Github | [https://github.com/liuruoze/mini-AlphaStar](https://github.com/liuruoze/mini-AlphaStar)
Gitee | [https://gitee.com/liuruoze/mini-AlphaStar](https://gitee.com/liuruoze/mini-AlphaStar)

## History

This is the introduction of previous versions of mAS here [HISTORY](doc/HISTORY.MD).

## Furture

There are still some todos (very few) that need to be filled up and improved.

## Usage

If you find some problems or questions, try to find here [USAGE](doc/USAGE.MD).

## Results

Here are some illustration figures of SL training process below:

![SL training process](doc/SL_traing.png)

## Citing

If you find this repository useful, please cite our project:
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

The technical report is now on arxiv named as [An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890).

We will give two to three updates for the report, to make it more complete and clear. 


## Thinking

We recommend the readers to refer to our paper to find some rethinking of AlphaStar in [Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452).


## Paper

We will give a paper which may be available in the future presenting the experiments and evaluations on using it. 