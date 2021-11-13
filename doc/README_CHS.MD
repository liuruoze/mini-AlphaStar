＃mini-AlphaStar

## 介绍

我们发布了mini-AlphaStar项目（v_0.X版），它是DeepMind原始AlphaStar程序的微型复现版本。

“v_0.X”表示我们认为已经实现了X * 10％以上的代码。

“mini”意味着我们使原始的AlphaStar超参数可调且微小化，以便可以小规模地运行。

在设计mAS时，我们参考了“奥卡姆剃刀原则”，简单就是好的。

因此我们从零开始构建mAS，除非对速度和性能影响很大的功能，能够不加就不加。

同时也尽可能不要使用过多的依赖包，让mAS最好只依赖于PyTorch（最多指定最低的版本）。

通过这样的方式，我们大大简化了mAS的学习成本，让mAS的架构和代码读起来相对容易。

另外，在代码中我们还添加了很多的注释，确保一些关键的设计能够传达给读者。

## 目录

下表显示了项目中的相应模块。

模块|内容
------------ | -------------
alphastarmini.core.arch | alphaStar体系结构
alphastarmini.core.sl | 监督学习
alphastarmini.core.rl | 强化学习
alphastarmini.core.ma | 多代理联赛训练
alphastarmini.lib | 相关库函数
alphastarmini.third | 第三方函数
res | 其它有用的资料

## 要求

Pytorch >= 1.5，其他请参阅requirements.txt。

## 地点

代码库被放置在以下位置：

位置 | 网址
------------ | -------------
Github | [https://github.com/liuruoze/mini-AlphaStar](https://github.com/liuruoze/mini-AlphaStar)
Gitee | [https://gitee.com/liuruoze/mini-AlphaStar](https://gitee.com/liuruoze/mini-AlphaStar)

## 展望

还有一些部分仍需要实现，例如，z的计算，vtrace的损失部分以及需要填充的env中的某些信息。

## 引用

如果您发现此存储库有用，请引用我们的项目：
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

## 报告

我们将在大约一个月的时间以内提交一份技术报告，以介绍其设计和使用细节。

## 论文

我们将提供一篇可能会在将来可见的论文，介绍使用它的实验和评估。