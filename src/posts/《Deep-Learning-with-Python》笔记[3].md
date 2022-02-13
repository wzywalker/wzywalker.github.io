---
layout: post
title: 《Deep Learning with Python》笔记[3]
slug: Deep-Learning-with-Python-Notes-3
date: 2021-09-18 17:25
status: publish
author: walker
categories: 
  - AI
tags:
  - deep learning
  - 深度学习
  - keras
  - cv
  - nlp
  - tensorflow
  - gan
  - lstm
  - language mode
  - rnn
  - heatmap
  - dropout
  - machine learning
---

# Fundamentals of machine learning

* Supervised learning
    * binary classification
    * multiclass classificaiton
    * scalar regression
    * vector regression（比如bounding-box)
    * Sequence generation (摘要，翻译...)
    * Syntax tree prediction
    * Object detection (一般bounding-box的坐标仍然是回归出来的)
    * Image segmentation
* Unsupervised learing
    * 是数据分析的基础，在监督学习前也常常需要用无监督学习来更好地“理解”数据集
    * 主要有降维(`Dimensionality reduction`)和聚类(`clustering`)
* Self-supervised learning
    * 其实还是监督学习，因为它仍需要与某个target做比较
    * 往往半监督（自监督）学习仍然有小量有标签数据集，在此基础上训练的不完善的model用来对无标签的数据进行打标，循环中对无标签数据打标的可靠度就越来越高，这样总体数据集的可靠度也越来越高了。有点像生成对抗网络里生成器和辨别器一同在训练过程中完善。
    * `autoencoders`
* Reinforcement learning
    * an `agent` receives information about its `environment` and learns to choose `actions` that will maximize some `reward`. 
    * 可以用训练狗来理解
    * 工业界的应用除了游戏就是机器人了

## Data preprocessing
* vectorization
* normalization (small, homogenous)
* handling missing values
    1. 除非0有特别的含义，不然一般可以对缺失值补0
    2. 你不能保证测试集没有缺失值，如果训练集没看到过缺失值，那么将不会学到忽略缺失值
        * *复制*一些训练数据并且随机drop掉一些特征
* feature extraction
    * making a problem easier by expressing it in a simpler way. It usually requires understanding the problem **in depth**.
    * **Before** deep learning, feature engineering used to be `critical`, because classical **shallow algorithms** didn’t have `hypothesis spaces` rich enough to learn useful features by themselves. (又见假设空间)
    * 但是好的特征仍然能让你在处理问题上更优雅、更省资源，也能减小对数据集规模的依赖。

## Overfitting and underfitting

* Machine learning is the tension between `optimization` and `generalization`.
* optimization要求你在训练过的数据集上能达到最好的效果
* generalization则希望你在没见过的数据上有好的效果
* 如果训练集上loss小，测试集上也小，说明还有优化(optimize)的余地 -> `underfitting`看loss
    * just keep training
* 如果验证集上generalization stop improving(泛化不再进步，一般看衡量指标，比如准确率) -> `overfitting`

解决overfitting的思路：

* **the best solution** is get more trainging data
* **the simple way** is to reduce the size of the model
    * 模型容量(`capacity`)足够大，就足够容易*记住*input和target的映射，没推理什么事了
* add constraints -> weight `regularization`
* add dropout
    
## Regularization

**Occam’s razor**

> given *two explanations* for something, the explanation most likely to be correct is the **simplest one**—the one that makes **fewer assumptions**. 

即为传说中*如无必要，勿增实体*的`奥卡姆剃刀原理`，这是在艺术创作领域的翻译，我们这里还是直译的好，即能解释一件事的各种理解中，越简单的，假设条件越少的，往往是最正确的，引申到机器学习，就是如何定义一个`simple model`

A simple model in this context is:

* a model where the distribution of parameter values has `less entropy` 
* or a model with fewer parameters

实操就是，就是迫使选择那些值比较小的weights，which makes the distribution of weight values more regular. This is called weight `regularization`。这个解释是我目前看到的最`regularization`这个名字最好的解释，“正则化”三个字都认识，根本没人知道这三个字是什么意思，翻译了跟没番一样，而使分布更“常规化，正规化”，好像更有解释性。

别的教材里还会告诉你这里是对大的权重的**惩罚**（设计损失函数加上自身权重后，权重越大，loss也就越大，这就是对大权重的惩罚）

* L1 regularization—The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights).
* L2 regularization—The cost added is proportional to the square of the value of the weight coefficients (the L2 norm of the weights). 

L2 regularization is also called `weight decay `in the context of neural networks. Don’t let the different name confuse you: weight decay is mathematically **the same as** L2 regularization.

> 只需要在训练时添加正则化

## Dropout

randomly dropping out (setting to zero) a number of output features of the layer during training.

dropout的作者Geoff Hinton解释dropout的灵感来源于银行办事出纳的不停更换和移动的防欺诈机制，可能认为一次欺诈的成功实施需要员工的配合，所以就尽量降低这种配合的可能性。于是他为了防止神经元也能聚在一起”密谋”，尝试随机去掉一些神经元。以及对输出添加噪声，让模型更难记住某些patten。

## The universal workflow of machine learning

1. Defining the problem and assembling a dataset
    * What will your input data be? 
    * What are you trying to predict?
    * What type of problem are you facing?
    * You hypothesize that your outputs can be predicted given your inputs.
    * You hypothesize that your available data is sufficiently informative to learn the relationship between inputs and outputs.
    * Just because you’ve assembled exam- ples of inputs X and targets Y doesn’t mean X contains enough information to predict Y.
2. Choosing a measure of success
    * accuracy? Precision and recall? Customer-retention rate?
    * balanced-classification problems,
        * accuracy and area under the `receiver operating characteristic curve` (ROC AUC) 
    * class-imbalanced problems
        * precision and recall. 
    * ranking problems or multilabel classification
        * mean average precision
    * ...
3. Deciding on an evaluation protocol
    * Maintaining a hold-out validation set—The way to go when you have plenty of data
    * Doing `K-fold` cross-validation—The right choice when you have too few samples for hold-out validation to be reliable
    * Doing `iterated K-fold` validation—For performing highly accurate model evaluation when *little data* is available
4. Preparing your data
    * tensor化，向量化，归一化等
    * may do some feature engineering
5. Developing a model that does better than a baseline
    * baseline:
        * 基本上是用纯随机(比如手写数字识别，随机猜测为10%)，和纯相关性推理（比如用前几天的温度预测今天的温度，因为温度变化是连续的），不用任何机器学习做出baseline
    * model:
        * Last-layer activation
            * sigmoid, relu系列， 等等
        * Loss function
            * 直接的预测值真值的差，如MSE
            * 度量代理，如crossentropy是ROC AUC的proxy metric
    * Optimization configuration
        * What optimizer will you use? What will its learning rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.
    * Scaling up: developing a model that overfits
        * 通过增加layers, 增加capacity，增加training epoch来加速overfitting，从而再通过减模型和加约束等优化
    * Regularizing your model and tuning your hyperparameters
        * Add dropout.
        * Try different architectures: add or remove layers. 
        * Add L1 and/or L2 regularization.
        * Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
        * Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

Problem type | Last-layer activation | Loss function
-------------|-----------------------|--------------
Binary classification | sigmoid | binary_crossentropy
Multiclass, single-label classification | softmax |categorical_crossentropy
Multiclass, multilabel classification | sigmoid | binary_crossentropy
Regression to arbitrary values | None | mse
Regression to values between 0 and 1 | sigmoi | mse or binary_crossentropy
