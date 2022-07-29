# mini-AlphaStar

## Introduction

The mini-AlphaStar (mini-AS, or mAS) project is a **mini-scale** version of the AlphaStar (AS) program. AlphaStar is the intelligent AI proposed by DeepMind to play StarCraft II. Note the mini-AS is a research project. It is not an official product of DeepMind.

The "mini-scale" means making the original AS's hyper-parameters **adjustable** so that we can train mini-AS on a **small** scale, e.g., in a single common commercial server machine.

We referred to the "Occam's Razor Principle" when designing the mini-AS: **simple** is good. Therefore, we build the mini-AS from scratch. Unless the function significantly impacts speed and performance, we shall omit it. Meanwhile, we also try not to use too many dependency packages so that mini-AS should only depend on the PyTorch. In this way, we simplify the learning cost of the mini-AS and make the architecture of mini-AS relatively **easy**. 

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

This release is the "v_1.08" version. The main changes are as follows:

* The training of supervised learning is improved, and the batch_size and sequence_length are different from those in reinforcement learning, which enhances the learning effect;
* For reinforcement learning, we use use_action_type_mask now, which improves learning efficiency;
* Added options to turn off entity_encoder and autoregressive_embedding for testing the impact of these modules;
* Fixed some known issues;
* A trained RL model is provided, which is first trained by the SL method and then be trained by the RL method;

## Hints

**Warning**: SC2 is extremely difficult, and AlphaStar is also very complex. Though our project is a mini-AlphaStar, it has almost the similar technologies as AS, and the training resource also costs very high. We can hardly train mini-AS on a laptop. The recommended way is to use a commercial server with a GPU card plus large memory and disk space. For someone interested in this project for the first time, we recommend you collect (star) this project. Devolve deeply into researching it only when you have enough free time and training resources.

## Location

The codebase and videos are hosted in two places. 

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

PyTorch >= 1.5. Please see requirements.txt for other dependencies.

## Install

The [SCRIPT Guide](scripts/Setup_cmd.MD) gives some commands to install PyTorch by conda (this will automatically install CUDA and cudnn, which is convenient).

E.g. like (to install PyTorch 1.5 with accompanied CUDA and cudnn):
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

After you have done all requirements, run the python file below:
```
python run.py
```
You may use comments and uncomments in "run.py" to select the training process you want.

The [USAGE Guide](doc/USAGE.MD) provides answers to some problems and questions.

You should follow the following instructions to get similar or better results than the provided gifs on the main page.

We summarised the usage sequences as the following:

1. Transform replays: download the replays for training, then use the script in mAS to transform the replays to trainable data;
2. Supervised learning: use the trainable data to supervise learning an initial model;
3. Evaluate SL model: the trained SL model should be evaluated on the RL environment to make sure it behaves right;
3. Reinforcement learning: use the trained SL model to do reinforcement learning in the SC environment, seeing the win rate starts growing.

We give detailed descriptions below.

## Transform replays

In supervised learning, you first need to download SC2 replays.

The [REPLAY Guide](doc/REPLAY.MD) shows a guide to download these SC2 replays.

The [ZHIHU Guide](https://zhuanlan.zhihu.com/p/410523216) provides Chinese users who are not convenient to use Battle.net (outside China) a guide to download replays.

After downloading replays, you should move the replays to "./data/Replays/filtered_replays_1" (you can change the name in `transform_replay_data.py`).

Then use `transform_replay_data.py` to transform these replays to pickles or tensors (you can change the output type in the code of that file).

You don't need to run the transform_replay_data.py directly. Only run "run.py" is OK. Make sure run.py has the following lines uncommented. Then you can directly run "run.py".

```
    # from alphastarmini.core.sl import transform_replay_data
    # transform_replay_data.test(on_server=P.on_server)
```

**Note**: To get the effect of the trained agent in the gifs, use the replays in [Useful-Big-Resources](https://github.com/liuruoze/Useful-Big-Resources/blob/main/replays). These replays are generatedy by our experts, to get an agent having the ability to win the built-in bot.

## Supervised learning

After getting the trainable data (we recommend using tensor dat). Make sure run.py has the following lines uncommented. Then you can directly run "run.py" to do supervised learning.

```
    # from alphastarmini.core.sl import sl_train_by_tensor
    # sl_train_by_tensor.test(on_server=P.on_server)
```

The default learning rate is 1e-4, and the optimal training epochs should be 10 (more epochs may cause the training effect overfitting).

From the v_1.05 version, we support multi-GPU supervised learning (not recommended now) training for mini-AS, improving the training speed. The way to use multi-GPU training is straightforward, as follows:
```
python run_multi-gpu.py
```

Multi-GPU training has some unstable factors (caused because of PyTorch). If you find your multi-GPU training has training instability errors, please switch to the single-GPU training.

We currently support four types of supervised training, which all reside in the "alphastarmini.core.sl" package.

File | Content
------------ | -------------
`sl_train_by_pickle.py` | pickle (data not preprocessed) training: Slow, but need small disk space.
`sl_train_by_tensor.py` | tensor (data preprocessed) training: Fast, but cost colossal disk space.
`sl_multi_gpu_by_pickle.py` | multi-GPU, pickle training: It has a requirement need for large shared memory.
`sl_multi_gpu_by_tensor.py` | multi-GPU, tensor training: It needs both large memory and large shared memory.

You can use the `load_pickle.py` to transform the generated pickles (in "./data/replay_data") to tensors (in "./data/replay_data_tensor").

From the v_1.06 version, we still recommend using single-GPU training. 

The newest training ways (e.g., in v_1.07) are still in the single GPU type due to multi-GPU training cost too much memory.

## Evaluate SL model

After getting the supervised learning model, we should test the model's performance in the SC2 environment. The reason is that there is a domain shift from the SL data to the RL environment. 

Make sure run.py has the following lines uncommented. Then you can directly run "run.py" to do an evaluation of the SL model.

```
    # from alphastarmini.core.rl import rl_eval_sl
    # rl_eval_sl.test(on_server=P.on_server)
```

The evaluation is similar to RL training, but the updating is closed. The running is also in single-thread, to make the randomness due to multi-thread not affect the evaluation.

## Reinforcement learning

After ensuring the supervised learning model is OK and suitable for RL training, we can do RL based on the learned supervised learning model. 

Make sure run.py has the following lines uncommented. Then you can directly run "run.py" to do reinforcement learning.

```
    # from alphastarmini.core.rl import rl_vs_inner_bot_mp
    # rl_vs_inner_bot_mp.test(on_server=P.on_server, replay_path=P.replay_path)
```

Note RL training uses a multi-process plus multi-thread manner (to accelerate the learning speed), so make sure to run these codes on a high-performance computer. 

E.g., we run 15 processes, and each has two actor threads and one learner thread. If your computer is not strong, reduce the parallel nums.

The learning rate should be low (below 1e-5 because you are training on an initially trained model). The training iterations should be as long as best (more training iterations can reduce the instability of RL training).

If you find the training result is not good as you imagine, please open an issue to ask us or discuss with us (though we can not make sure to respond to it in time or there is any solution to every problem).

## Trained Models

You can find the trained models [here](https://github.com/liuruoze/Useful-Big-Resources/blob/main/mAS-models).

The "rl_22-02-07_16-26-48.pth" is the trained model by our method. On our server, it was first trained by supervised learning, and then it was trained by reinforcement learning for one day.

## Results

Here are some illustration figures of the SL training process below:

![SL training process](doc/SL_traing.png)

We can see the loss (one primary loss and six argument losses) fall quickly.

We provide more curves (like the accuracy curve) for the SL training process after the v_1.05 version.

The trained behavior of the agents shows in the gifs on this page.

Our later paper will provide a more detailed illustration of the experiments (such as the effects of different hyper-parameters).

## History

The [HISTORY](doc/HISTORY.MD) is the historical introduction of the previous versions of mini-AS.

## Citing

If you find our repository useful, please cite our project or the below technical report:
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

The [An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890) is a technical report introducing the mini-AS (not full version).
```
@article{liu2021mASreport,
  author    = {Ruo{-}Ze Liu and
               Wenhai Wang and
               Yanjie Shen and
               Zhiqi Li and
               Yang Yu and
               Tong Lu},
  title     = {An Introduction of mini-AlphaStar},
  journal   = {CoRR},
  volume    = {abs/2104.06890},
  year      = {2021},
}
```

## Rethinking

The [Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452) is our thinking of the advantages and disadvantages of AlphaStar.


## Paper

We will give a paper (which is now under peer-review) that may be available in the future, presenting detailed experiments and evaluations using the mini-AS. 
