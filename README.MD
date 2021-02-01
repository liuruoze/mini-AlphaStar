# mini-AlphaStar

## Introduction

We release the mini-AlphaStar project (v_0.7), which is a mini source version of the original AlphaStar program by DeepMind. 

"v_0.7" means we think we have implemented above 70 percent code of it. 

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
Gitee | [https://gitee.com/easypr/mini-AlphaStar](https://gitee.com/easypr/mini-AlphaStar)

## Furture

There are some parts that still need to be fulfilled, e.g., the calculation of z, the loss part of the vtrace, and some missing information of the env to be filled up.

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

## Paper

We will give a paper which may be available in the future presenting the experiments and evaluations on using it. 