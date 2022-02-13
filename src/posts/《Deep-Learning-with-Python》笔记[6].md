---
layout: post
title: 《Deep Learning with Python》笔记[6]
slug: Deep-Learning-with-Python-Notes-6
date: 2021-10-03 18:29
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

# Advanced deep-learning best practices

这一章是介绍了更多的网络（从keras的封装特性出发）结构和模块，以及batch normalization, model ensembling等知识。

## beyond Sequential model

前面介绍的都是Sequential模型，就是一个接一个地layer前后堆叠，现实中有很多场景并不是一进一出的：

1. multi-input model

假设为二手衣物估价：
* 格式化的元数据（品牌，性别，年龄，款式）: one-hot, dense
* 商品的文字描述：RNN or 1D convnet
* 图片展示：2D convnet
* 每个input用适合自己的网络做输出，然后合并起来作为一个input，回归一个价格

2. multi-output model (multi-head)

一般的检测器通常就是多头模型，因为既要回归对象类别，还要回归出对象的位置

3. graph-like model

这个名字很好地形容了做深度学习时看别人的网络是什么样的方式：看图。现代的SOTA的网络往往既深且复杂，而网络结构画出来也不再是一条线或几个简单分支，这本书干脆把它们叫图形网络：`Inception`, `Residual`

为了能架构这些复杂的网络，keras介绍了新的语法，先看看怎么重写`Sequential`:

```python
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# 重写
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)

model.summayr()
```

我们自己实现过静态图，最终去执行的时候能从尾追溯到头，并从头来开始计算，这里也是一样的：
1. input, output是Tensor类，所以有完整的层次信息
2. output往上追溯，最终溯到缺少一个input
3. 这个input恰好也是Model的构造函数之一，闭环了。

书里说的更简单，output是input不断transforming的结果。如果传一个没有这个关系的input进去，就会报错。

**demo**

用一个QA的例子来演示多输入（一个问句，一段资料），输出为答案在资料时的索引（简化为单个词，所以只有一个输出）

```python
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(
    64, text_vocabulary_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)  # lstm 处理资讯
question_input = Input(shape=(None,), dtype='int32', name='question')


embedded_question = layers.Embedding(
    32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question) # lstm 处理问句

concatenated = layers.concatenate([encoded_text, encoded_question], axis = -1)  # 竖向拼接（即不增加内容只增加数量）
answer = layers.Dense(answer_vocabulary_size,
                      activation='softmax')(concatenated)
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
```

这里是把答案直接给回归出来了(one-hot)，如果是给出答案的首尾位置，那肯定只能用索引了。

**demo**

多头输出的：

```python
# 线性回归
age_prediction = layers.Dense(1, name='age')(x)
# 逻辑回归
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
# 二元逻辑回归
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = Model(posts_input,
              [age_prediction, income_prediction, gender_prediction])
```

梯度回归要求loss是一个标量，keras提供了方法将三个loss加起来，同时为了量纲统一，还给了权重参数：
```python
model.compile(optimizer='rmsprop',
loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[0.25, 1., 10.])
```

## Directed acyclic graphs of layers

有向无环图。可以理解为最终不会回到出发点。

现在会介绍的是几个`Modules`，意思是可以把它当成一个layer，来构造你的网络/模型。

### Inception Modules

* inspired by `network-in-network`
* 对同一个输入做不同（层数/深度）的卷积（保证最终相同的下采样维度），最后合并为一个输出
* 因为卷积的深度不尽相同，学到的空间特征也有粗有细

### Residual Connections

* 有些地方叫shortcut
* 用的是相加，不是concatenate, 如果形状变了，对earlier activation做linear transformation
* 解决`vanishing gradients` and `representational bottlenecks`
* adding residual connections to any model that has more than 10 layers is likely to be beneficial.

**representational bottlenecks**

序列模型时，每一层的表示都来自于前一层，如果前一层很小，比如维度过低，那么携带的信息量也被压缩得很有限了，整个模型都会被这个“瓶颈”限制。比如音频信号处理，降维就是降频，比如到0-15kHz，但是下游任务也没法recover dropped frequencies了。所有的损失都是永久的。

Residual connections, by `reinjecting` earlier information downstream, partially solve this issue for deep-learning models.（又一次强调`reinject`）

### Lyaer weight sharihng

在网络的不同位置用同一个layer，并且参数也相同。等于共享了相同的知识，相同的表示，以及是同时(simultaneously)训练的。

一个语义相似度的例子，输入是A和B还是B和A，是一样的（即可以互换）。架构网络的时候，用LSTM来处理句子，需要做两个LSTM吗？当然可以，但是也可以只做一个LSTM，分别喂入两个句子，合并两个输出来做分类。就是考虑到这种互换性，既然能互换，也就是这个layer也能应用另一个句子，因此就不必要再新建一个LSTM.

### Models as layers

讲了两点：
1. model也可以当layer使用
2. 多处使用同一个model也是共享参数，如上一节。

举了个双摄像头用以感知深度的例子，每个摄像头都用一个Xception网络提取特征，但是可以共用这个网络，因为拍的是同样的内容，只需要处理两个摄像头拍到的内容的差别就能学习到深度信息。因为希望是用同样的特征提取机制的。

都是蜻蜓点水。

## More Advanced

### Batch Normalization

1. 第一句话就是说为了让样本数据看起来**更相似**，说明这是初衷。
2. 然后是能更好地泛化到未知数据（同样也是因为bn后就**更相似**了）
3. 深度网络中每一层之后也需要做
    * 还有一个书里没讲到的原因，就是把值移到激活函数的梯度大的区域（比如0附近），否则过大过小的值在激活函数的曲线里都是几乎没有梯度的位置
4. 内部用的指数移动平均(`exponential moving average`)
5. 一些层数非常深的网络必须用BN，像resnet 50, 101, 152, inception v3, xception等

### Depthwise Separable Convolution

之前的卷积，不管有多少个layer，都是放到矩阵里一次计算的，DSC把每一个layer拆开，单独做卷积（不共享参数），因为没有一个巨大的矩阵，变成了几个小矩阵乘法，参数量也大大变少了。

1. 对于小样本很有效
2. 对于大规模数据集，它可以成为里面的固定结构的模块（它也是Xception的基础架构之一）

> In the future, it’s likely that depthwise separable convolutions will `completely replace regular convolutions`, whether for 1D, 2D, or 3D applications, due to their higher representational efficiency.

?!!

### Model ensembling

1. Ensembling consists of **pooling together** the predictions of a set of different models, to produce better predictions.
2. 期望每一个`good model`拥有`part of the truth`(部分的真相)。盲人摸象的例子，没有哪个盲人拥有直接感知一头象的能力，机器学习可能就是这样一个盲人。
3. The key to making ensembling work is the `diversity` of the set of classifiers -> 关键是要“多样性”。 `Diversity` is what makes ensembling work.
4. 千万**不要**去ensembling同样的网络仅仅改变初始化而去train多次的结果。
5. 比较好的实践有ensemble `tree-based` models(random forests, gradient-boosted trees) 和深度神经网络
6. 以及`wide and deep` category of models, blending deep learning with shallow learning. 

同样是蜻蜓点水。
