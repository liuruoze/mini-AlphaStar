# mini-AlphaStar


## Update

This is the "v_0.93" version, and the updates are as follows:

* Add the implement the unit_counts_bow;
* Add the implementation of the build_order;
* Implement upgrade, effect, last and available action, plus time coding;
* Add reward calculation for build order and unit counts
* Add the calculation of td_lambda_loss based on the reward of the Levenstein and Hamming;

## Introduction

We release the mini-AlphaStar project, which is a mini source version of the original AlphaStar program. AlphaStar is an intelligent AI proposed by DeepMind to play StarCraft II. StarCraft II is an RTS game developed by Blizzard.

"v_0.X" means we think we have implemented above X * 10 percent code of it. 

"mini" means that we make the original AlphaStar hyperparameter adjustable so that it can run on a small scale.

The readme for the Chinese version is at [here](README_CHS.MD).

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

## Furture

There are still some parts that need to be fulfilled, e.g., the calculation of z, to be filled up.

## Citing

If you find this repository useful, please cite our project:
```
@misc{mini-AlphaStar,
  author = {Ruo{-}Ze Liu and Wenhai Wang and Yang Yu and Tong Lu},
  title = {mini-AlphaStar},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuruoze/mini-AlphaStar}},
}
```

## Report

We will present a technical report to introduce the design and usage details of it in about one month. 

The technical report is now on arxiv named as [An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890).

If you find this report useful, please cite the report:
```
@misc{report_mini-AlphaStar,
      title={An Introduction of mini-AlphaStar}, 
      author={Ruo-Ze Liu and Wenhai Wang and Yanjie Shen and Zhiqi Li and Yang Yu and Tong Lu},
      year={2021},
      journal={CoRR},
      eprint={2104.06890},
      archivePrefix={arXiv},
}
```

## Paper

We will give a paper which may be available in the future presenting the experiments and evaluations on using it. 