# mini-AlphaStar


## Update

This is the "v_0.99" version, and the updates are as follows:

* Add the code for analyzing the statistics of replays;
* Implement the RL training with the z reward (from the experts' statistics in replays);
* Replenish the entropy_loss_for_all_arguments;
* True implementation of teacher_logits in human_policy_kl_loss and test all the above RL training on server;
* Decouple the sl_loss with the agent, and pass the test of SL training on the server;
* Fix some warnings due to PyTorch 1.5, and fill up many TODOs;

## Introduction

We release the mini-AlphaStar project, which is a mini source version of the original AlphaStar program. AlphaStar is an intelligent AI proposed by DeepMind to play StarCraft II. StarCraft II is an RTS game developed by Blizzard.

"v_0.X" means we think we have implemented above X % percent code of it. 

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

There are still some todos (very few) that need to be filled up and improved.

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

The technical report is now on arxiv named as [An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890).

We will give two to three updates for the report, to make it more complete and clear. 

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