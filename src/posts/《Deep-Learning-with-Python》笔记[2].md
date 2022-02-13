---
layout: post
title: 《Deep Learning with Python》笔记[2]
slug: Deep-Learning-with-Python-Notes-2
date: 2021-09-15 00:00
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

# Getting started with neural networks

## Anatomy of a neural network

* `Layers`, which are combined into a `network` (or model)
    * layers: 常见的比如卷积层，池化层，全连接层等
    * models: layers构成的网络，或多个layers构成的模块（用模块组成网络）
        * Two-branch networks
        * Multihead networks
        * Inception blocks, residual blocks etc.
    * The topology of a network defines a hypothesis space
    * 本书反复强调的就是这个`hypothesis space`，一定要理解这个思维：
        * By choosing a network topology, you `constrain` your space of possibilities (hypothesis space) to a specific series of tensor operations, mapping input data to output data.（network的选择约束了tensor变换的步骤）
        * 所以如果选择了不好的network，可能导致你在错误的`hyposhesis space`里搜索，以致于效果不好。
* The `input data` and corresponding `targets`
* The `loss` function (objective function), which defines the `feedback signal` used for learning
    * The quantity that will be minimized during training. 
    * It represents a measure of success for the task at hand.
    * 多头网络有多个loss function，但基于`gradient-descent`的网络只允许有一个标量的loss，因此需要把它合并起来（相加，平均...）
* The `optimizer`, which determines how learning proceeds
    * Determines how the network will be updated based on the loss function. 
    * It implements a specific variant of stochastic gradient descent (SGD).

### Classifying movie reviews: a binary classification example

**一个二元分类的例子**

情感分析/情绪判断，数据源是IMDB的影评数据.

**理解hidden的维度** 

how much freedom you’re allowing the network to have when learning internal representations. 即学习表示（别的地方通常叫提取特征）的自由度。

目前提出了架构网络的时候的两个问题：

1. 多少个隐层
2. 隐层需要多少个神经元（即维度）

后面的章节会介绍一些原则。

**激活函数**

李宏毅的课程里，从用整流函数来逼近非线性方程的方式来引入激活函数，也就是说在李宏毅的课程里，激活函数是**因**，推出来的公式是**果**，当然一般的教材都不是这个角度，都是有了线性方程，再去告诉你，这样还不够，需要一个`activation`。

本书也一样，告诉你，如果只有`wX+b`，那么只有线性变换，这样会导致对`hypothesis space`的极大的限制，为了扩展它的空间，就引入了非线性的后续处理。总之，都是在自己的逻辑体系内的。本书的逻辑体系就是`hypothesis space`，你想要有解，就是在这个空间里。

**网络结构**

```python
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

**entropy**

`Crossentropy` is a quantity from the field of Information Theory（信息论） that measures the distance between probability distributions。

in this case, between the ground-truth distribution and your predictions.

**keras风格的训练**

其实就是模仿了`scikit learn`的风格。对快速实验非常友好，缺点就是封装过于严重，不利于调试，但这其实不是问题，谁也不会只用keras。

```python
# 演示用类名和字符串分别做参数的方式
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])

from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# train
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```

后续优化，就是对比train和validate阶段的loss和accuracy，找到overfit的节点（比如是第N轮），然后重新训练到第N轮（或者直接用第N轮生成的模型，如果有），用这个模型来预测没有人工标注的数据。

核心就是要**训练到明显的overfit**为止。这是第一个例子的内容，所以是告诉你怎么用这个简单的网络来进行预测，而不是立即着眼怎么去解决overfit.

**第一个小结**

1. 数据需要预处理成tensor, 了解几种tensor化，或vector化的方式
2. 堆叠全连接网络(Dense)，以及activation，就能解决很多分类问题
3. 二元分类的问题通常在Dense后接一个sigmoid函数
4. 引入二元交叉熵(BCE)作为二元分类问题的loss
5. 用了rmsprop优化器，暂时没有过多介绍。这些优化器都是为了解决能不能找到局部极值而进行的努力，具体可看上一篇李宏毅的笔记
6. 使用overfit之前的那一个模型来做预测

### Classifying newswires: a multiclass classification example

这次用路透社的新闻来做多分类的例子，给每篇新闻标记类别。

**预处理，一些要点**:

1. 不会采用所有的词汇，所以预处理时，根据词频，只选了前1000个词
2. 用索引来实现文字-数字的对应
3. 用one-hot来实现数字-向量的对应
4. 理解什么是序列（其实就是一句话）
5. 所以句子有长有短，为了矩阵的批量计算（即多个句子同时处理），需要“对齐”（补0和截断）
6. 理解稠密矩阵(word-embedding)与稀疏矩阵(one-hot)的区别（这里没有讲，用的是one-hot)

**网络和训练**

1. 网络结构不变，每层的神经元为(64, 64, 46)
2. 前面增加了神经元，16个特征对语言来说应该是不够的）
3. 最后一层由1变成了46，因为二元的输出只需要一个数字，而多元输出是用one-hot表示的向量，最有可能的类别在这个向量里拥有最大的值。
4。 损失函数为`categorial_crossentropy`，这在别的教材里应该就是普通的CE.

**新知识**

1. 介绍了一种不用one-hot而直接用数字表示真值的方法，但是没有改变网络结构（即最后一层仍然输出46维，而不是因为你用了一个标量而只输出一维。
    * 看来它仅仅就是一个**语法糖**（loss函数选择`sparse_categorial_crossentropy`就行了）
2. 尝试把第2层由64改为4，变成`bottleneck`，演示你有46维的数据要输出的话，前面的层数或少会造成信息压缩过于严重以致于丢失特征。

### Predicting house prices: a regression example

这里用了预测房价的Boston Hosing Price数据集。

与吴恩达的课程一样，也恰好是在这个例子里引入了对input的normalize，理由也仅仅是简单的把量纲拉平。现在我们应该还知道Normalize还能让数据在进入激活函数前，把值限定在激活函数的梯度敏感区。

此外，一个知识点就是你对训练集进行Normalize用的均值和标准差，是直接用在测试集上的，而不是各计算各的，可以理解为保持训练集的“分布”。
> 这也是`scikit learn`里`fit_tranform`和直接用`transform`的原因。

1. 对scalar进行预测是不需要进行激活（即无需把输出压缩到和为1的概率空间）
2. loss也直观很多，就是predict与target的差（取平方，除2，除批量等都是辅助），预测与直值的差才是核心。