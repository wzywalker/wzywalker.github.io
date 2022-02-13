---
layout: post
title: 《Deep Learning with Python》笔记[1]
slug: Deep-Learning-with-Python-Notes-1
date: 2021-09-12 00:00
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

本来是打算趁这个时间好好看看花书的，前几章看下来确实觉得获益匪浅，但看下去就发现跟不上了，特别是抱着急功近利的心态的话，目前也沉不下去真的一节节吃透地往下看。这类书终归不是入门教材，是需要你有过一定的积累后再回过头来看的。

于是想到了《Deep Learning with Python》，忘记这本书怎么来的了，但是在别的地方看到了有人推荐，说是Keras的作者写的非常好的一本入门书，翻了前面几十页后发现居然跟进去了，不该讲的地方没讲比如数学细节，而且思路也极其统一，从头贯穿到尾（比如representations, latent space,  hypothesis space），我觉得很受用。

三百多页全英文，居然也没查几个单词就这么看完了，以前看文档最多十来页，也算一个突破了，可见其实还是一个耐心的问题。

看完后书上做了很多笔记，于是顺着笔记读了第二遍，顺便就把笔记给电子化了。不是教程，不是导读。

# Fundamentals of deep learning

**核心思想**：
learng useful `representations` of input data
>what’s a `representation`? 
>
>At its core, it’s a different way to look at data—to represent or encode data. 

简单回顾深度学习之于人工智能的历史，每本书都会写，但每本书里都有作者自己的侧重：
* Artificial intelligence
* Machine learning
    * Machine learning is tightly related to `mathematical statistics`, but it differs from statistics in several important ways. 
        * machine learning tends to deal with large, complex datasets (such as a dataset of millions of images, each consisting of tens of thousands of pixels) 
        * classical statistical analysis such as Bayesian analysis would be impractical(不切实际的). 
        * It’s a hands-on discipline in which ideas are proven empirically more often than theoretically.（工程/实践大于理论）
    * 是一种meaningfully transform data
        * Machine-learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amenable to the task at hand, such as a classification task.
        * 寻找更有代表性的representation, 通过:(coordinate change, linear projections, tranlsations, nonlinear operations)
        * 只会在`hypothesis space`里寻找
        * 以某种反馈为信号作为优化指导
* Deep learning
    * Machine Learing的子集，一种新的learning representation的新方法
    * 虽然叫神经网络(`neural network`)，但它既非neural，也不是network，更合理的名字：
        * `layered representations learning` and `hierarchical representations learning`.
    * 相对少的层数的实现叫`shallow learning`

## Before deep learning

* Probabilistic modeling
    *  the earliest forms of machine learning, 
    * still widely used to this day. 
        * One of the best-known algorithms in this category
 is the `Naive Bayes algorithm`(朴素贝叶斯)
    * 条件概率，把规则理解为“条件”，判断概率，比如垃圾邮件。
        * A closely related model is the logistic regression
* Early neural networks
    * in the mid-1980s, multiple people independently rediscovered the Backpropagation algorithm
    * The `first` successful practical application of neural nets came in 1989 from Bell Labs -> **LeNet**
* Kernel methods
    *  Kernel methods are `a group of classification algorithms`(核方法是一组分类算法)
        * the best known of which is the `support vector machine` (**SVM**).
        * SVMs aim at solving classification problems **by** finding good *decision boundaries* between two sets of points belonging to two different categories.
            1. 先把数据映射到高维，decision boundary表示为`hyperplane`
            2. 最大化每个类别里离hyperplane最近的点到hyperplane的距离:`maximizing the margin`
        * The technique of mapping data to a high-dimensional representation 非常消耗计算资源，实际使用的是核函数(`kernel function`):
            * 不把每个点转换到高维，而只是计算每两个点在高维中的距离
            * 核函数是手工设计的，不是学习的
        * SVM在分类问题上是经典方案，但难以扩展到大型数据集上
        * 对于perceptual problems(感知类的问题)如图像分类效果也不好
            * 它是一个`shallow method`
            * 需要事先手动提取有用特征(`feature enginerring`)-> difficult and  brittle（脆弱的）
* Decision trees, random forests, and gradient boosting machines
    * Random Forest
        * you could say that they’re almost always the *second-best* algorithm for any shallow machine-learning task. 
    * gradient boosting machines (1st):
        * a way to improve any machine-learning model by iteratively training new models that specialize in `addressing the weak points of the previous models`.

## What makes deep learning different

it completely automates what *used to be* **the most crucial step** in a machine-learning workflow: `feature engineering`. 有人认为这叫穷举，思路上有点像，至少得到特征的过程不是靠观察和分析。

**feature engineering**
> manually engineer good layers of representations for their data