# mini-AlphaStar

## Introduction

We release the mini-AlphaStar project (v_0.7), which is a mini source version of the original AlphaStar program by DeepMind. 

"v_0.7" means we think we have implemented above 70 percent code of it. 

"mini" means that we make the original AlphaStar hyperparameter adjustable so that it can run on a small scale.

## Contents

The below table shows the majorites of the project.

Packages | Content
------------ | -------------
alphastarmini.core.arch | codes for the AlphaStar Architecture
alphastarmini.core.sl | codes for Surpervised Learning in it
alphastarmini.core.rl | codes for Reinforcement Learning in it
alphastarmini.core.ma | codes for Multi-Agent League Traning in it

## Requirements

Pytorch >= 1.5, others please see requirements.txt.

## Citing

If you find this repository useful, please cite our project:
```
@misc{mini-AlphaStar,
  author = {Ruo-Ze Liu, Wenhai Wang, Yang Yu, and Tong Lu},
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