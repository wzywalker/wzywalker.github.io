---
layout: post
title: 李宏毅MACHINE-LEARNING-2021-SPRING笔记
slug: 李宏毅MACHINE-LEARNING-2021-SPRING笔记
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - AI
---

> 纯听课时一些思路和笔记，没有教程作用。
> 这个课程后面就比较水了，大量的全是介绍性的东西，也罗列了大量的既往课程和论文，如果你在工作过研究中碰到了它提过的场景或问题，倒是可以把它作索引用。

# Linear Model

## Piecewise Linear
线性模型永远只有一条直线，那么对于折线（曲线），能怎样更好地建模呢？这里考虑一种方法，
1. 用一个`常数`加一个`整流函数`
    * 即左右两个阈值外y值不随x值变化，阈值内才是线性变化（天生就拥有两个折角）。
2. 每一个转折点加一个新的整流函数

如下：
![](../assets/1859625-8fa16cf5a344e174.png)

如果是曲线，也可以近似地理解为数个折线构成的（取决于近似的精度），而蓝色的整流函数不好表示，事实上有sigmoid函数与它非常接近（它是曲线），所以蓝线又可以叫：`Hard Sigmoid`

所以，最终成了一个常数(`bias`)和数个`sigmoid`函数来逼近真实的曲线。同时，每一个转折点`i`上s函数的具体形状（比如有多斜多高），就由一个新的线性变换来控制：$b_i + w_ix_n$，把`i`上**累积的线性变换**累加，就得到与$x_n$最可能逼近的曲线。

下图演示了3个转折点的情况：

![](../assets/1859625-9fd947d54a144ccf.png)

至此，一个简单的对b,w依赖的函数变成了对（$w_i, b_i, c_i$)和, x, b的依赖，即多了很多变量。

* $y = b + wx_1$
* $y = b + \sum_i c_i sigmoid(b_i + w_i x_{\color{red} 1})$ 

注意这个$x_1$，即只转了一个x就要堆一个`sum`，而目前也只是演示了只有一个特征的情况。

如果更复杂一点的模型，每次不是看一个x，而看n个x，（比如利用前7天的观看数据来预测第8天的，那么建模的时候就是每一个数都要与前7天的数据建立w和b的关系）：

> 其实就是由一个feature变成了n个feature了，一般的教材会用不同的feature来讲解（比如影响房价的除了时间，还有面积，地段等等），而这里只是增加了天数，可能会让人没有立刻弄清楚两者其实是同一个东西。其实就是x1, x2, x3...不管它们对应的是同一**类**特征，而是完全不同的多个**角度**的特征。

现在就有一堆$wx$了

* $y = b + \sum_j w_j x_j$
* 现在就变成了(注意，其实就是把加号右边完整代入）：
* $y = b + \sum_i c_i sigmoid(b_i + \color{red}{\sum_j w_{ij} x_j})$

展开计算，再根据特征，又可以看回矩阵了（而不是从矩阵出发来思考）：

![](../assets/1859625-8f33e9949d1295aa.png)

矩阵运算结果为(r)，再sigmoid后，设结果为a:

* $a_i = c_i \sigma(r_i)$
* $y = b + \sum_i a_i$ c 和 a要乘加，仍然可以矩阵化（其实是向量化）：
* $y = b + c^T a$， 把上面的展开回去：
* $y = b + c^T \sigma(\bold b + W x)$ 
    * 前后两个b是不同的，一个是数值，一个是向量

这里，我们把目前所有的“未知数”全部拉平拼成了一个向量 $\theta$：
![](../assets/1859625-c9eb59a590e5ae82.png)


这里，如果把$c^T$写成`W'`你会发现，我们已经推导出了一个2层的神经网络：一个隐层，一个输出层：
* b+wx 是第一层 得到`a`
* 对`a`进行一次sigmoid（别的教材里会说是激活）得到`a'`
* 把`a'`当作输入，再进行一次 b+wx (这就是隐层了)
* 得到的输出就是网络的输出`o`

> 这里在用另一个角度来尝试解释神经网络，激活函数等，但要注意，sigmoid的引入原本是去”对着折线描“的，也就是说是人为选择的，而这里仍然变成了机器去”学习“，即没有告诉它哪些地方是转折点。也就是说有点陷入了用机器学习解释机器学习的情况。

> 但是如果是纯曲线，那么其实是可以无数个sigmoid来组合的，就不存在要去拟合某些“特定的点”，那样只要找到最合适“数量”的sigmoig就行了（因为任何一个点都可以算是折点）

## Loss

loss 没什么变化，仍旧是一堆$\theta$代入后求的值与y的差，求和。并期望找到使loss最小化的$\theta$：

$\bold \theta = arg\ \underset{\theta}{min}\ L$

# Optimization

真实世界训练样本会很大，
* 我们往往不会把整个所有数据直接算一次loss，来迭代梯度，
* 而是分成很多小份(mini-batch)每一小份计算一次loss（然后迭代梯度）
* 下一个小batch认前一次迭代的结果
* 也就是说，其实这是一个不严谨的迭代，用别人数据的结果来当成本轮数据的前提
    * 最准确的当然是所有数据计算梯度和迭代。
    * 一定要找补的话，可以这么认为：
        * 即使一个小batch，也是可以训练到合理的参数的
        * 所以前一个batch训练出来的数据，是一定程度上合理的
        * 现在换了新的数据，但保持上一轮的参数，反而可以防止`过拟合`

![](../assets/1859625-9d953fa0a5c68501.png)

minibatch还有一个极端就是batchsize=1，即每次看完一条数据就与真值做loss，这当然是可以的，而且它非常快。但是：
1. 小batch虽然快，但是它非常noisy（及每一笔数据都有可能是个例，没有其它数据来抵消它的影响）
2. 因为有gpu平行运算的原因，只要不是batch非常大（比如10000以上），其实mini-batch并不慢
3. 如果是小样本，mini-batch反而更快，因为它一来可以平行运算，在计算gradient的时候不比小batch慢，但是它比小batch要小几个数量级的update.

仍然有个但是：实验证明小的batch size会有更高的准确率。
![](../assets/1859625-343cae0915fcaaa8.png)

两个local minimal，右边那个认为是不好的，因为它只要有一点偏差，与真值就会有巨大的差异。但是没懂为什么大的batch会更容易落在右边。

这是什么问题？其实是optimization的问题，后面会用一些方法来解决。

## Sigmoid -> RelU

前面我们用了soft的折线来模拟折线，其实还可以叠加两个真的折线(`ReLU`)，这才是我一直说的`整流函数`的名字的由来。

![](../assets/1859625-4479e5576c2ed5a0.png)

仔细看图，c和c'在第二个转折的右边，一个是向无穷大变，一个是向无穷小变，只要找到合理的斜率，就能抵消掉两个趋势，变成一条直线。

如果要用ReLU，那么简单替换一下： 

* $y = b + \sum_i {\color{ccdd00}{c_i}} sigmoid(\color{green}{b_i} + \sum_j \color{blue}{w_{ij}} x_j)$
* $y = b + \sum_{\color{red}2i} {\color{ccdd00}{c_i}} \color{red}{max}(\color{red}0,\ \color{green}{b_i} + \sum_j \color{blue}{w_{ij}} x_j)$

红色的即为改动的部分，也呼应了2个relu才构成一个sigmoid的铺垫。

把每一个a当成之前的x，我们可以继续套上新的w,b,c等，生成新的a->a'
![](../assets/1859625-ef692679760b967f.png)

![](../assets/1859625-3a13832b5c1c6b04.png)

而如果再叠一层，在课程里的资料里，在训练集上loss仍然能下降（到0.1），但是在测试集里，loss反而上升了（0.44)，这意味着开始过拟合了。

这就是反向介绍神经元和神经网络。先介绍数学上的动机，组成网络后再告诉你这是什么，而不是一上来就给你扯什么是神经元什么是神经网络，再来解释每一个神经元干了什么。

而传统的神经网络课程里，sigmoid是在逻辑回归里才引入的，是为了把输出限定在1和0之间。显然这里的目的不是这样的，是为了用足够多的sigmoid或relu来逼近真实的曲线（折线）

## Framework of ML

### 通用步骤：
1. 设定一个函数来描述问题$y = f_\theta(x)$, 其中$\theta$就是所有未知数（参数）
2. 设定一个损失函数$L(\theta)$
3. 求让损失函数尽可能小的$\theta^* = arg\ \underset{\theta}{\rm min}L(\theta)$

### 拟合不了的原因：
1. 过大的loss通常“暗示”了模型不合适（**model bias**），比如上面的用前1天数据预测后一天，可以尝试改成前7天，前30天等。
    * 大海里捞针，针其实不在海里
2. 优化问题，梯度下降不到目标值
    * 针在大海里，我却没有办法把它找出来

### 如何判断是loss optimization没做好？
用不同模型来比较（更简单的，更浅的）
![](../assets/1859625-a47194566126e9d4.png)

上图中，为什么56层的表现还不如20层呢？是`overfitting`吗？**不一定**。

我们看一下在训练集里的表现，56层居然也不如20层，这合理吗？ **不合理**

> 但凡20层能做到的，多出的36层可以直接全部identity（即复制前一层的输出），也不可能比20层更差（神经网络总可以学到的）

这时，就是你的loss optimization有问题了。

### 如何解决overfitting

1. 增加数据量
    * 增加数据量的绝对数量
    * data augmentation数据增强（比如反复随机从训练集里取，或者对图像进行旋转缩放位移和裁剪等）
3. 缩减模型弹性
    * （低次啊，更少的参数「特征」啊）
    * 更少的神经元，层数啊
    * 考虑共用参数
    * early stopping
    * regularization 
        * 让损失函数与每个特征系数直接挂勾，就变成了惩罚项
        * 因为它的值越大，会让损失函数越大，这样可以“惩罚”过大的权重
    * dropout
        * 随机丢弃一些计算结果

## Missmatch

课上一个测试，预测2/26的观看人数（周五，历史数据都是观看量低），但因为公开了这个测试，引起很多人疯狂点击，结果造成了这一天的预测结果非常差。

这个不叫overfitting，而是`mismatch`，表示的是**训练集和测试集的分布是不一样的**

mismatch的问题，再怎么增加数据也是不可能解决的。

## optimization problems

到目前为止，有两个问题没有得到解决：
1. loss optimization有问题怎么解决
    * 其实就是判断是不是saddle point（鞍点）
2. mismatch怎么解决

### saddle point
![](../assets/1859625-153bf93ab74dd8b4.png)

hessian矩阵是二次微分，当一次微分为0的时候，二次微分并不一定为0。这是题眼。 

对于红杠内的部分，设$\theta - \theta^T = v$，有：
* for all v: $v^T H v > 0 \rightarrow \theta'$附近的$\theta$都要更大
    * -> 确实是在`local minima`
* for all v: $v^T H v < 0 \rightarrow \theta'$附近的$\theta$都要更小
    * -> 确实是在`local maxima`
* 而时大时小，说明是在`saddle point`

事实上我们不可能去检查`所有的v`，这里用Hessian matrix来判断：
* $\rm H$ is `positive definite` $\rightarrow$ all eigen values are positive $\rightarrow$ local minimal
* $\rm H$ is `negative definite` $\rightarrow$ all eigen values are negative $\rightarrow$ local maximal

用一个很垃圾的网络举例，输入是1，输出是1，有w1, w2两层网络参数，因为函数简单，两次微分得到的hessian矩阵还是比较简单直观的：
![](../assets/1859625-43de51903c56d851.png)

由于特征值有正有负，我们判断在些(0, 0)这个`critical point`，它是一个`saddle point`.

如果你判断出当前的参数确实卡在了鞍点，它同时也指明了`update direction`!

![](../assets/1859625-162e0908ac2ce4d4.png)

图中，
1. 先构建出了一个小于0的结果，以便找到可以让$L(\theta)$收敛的目标
2. 这个结果依赖于找到这样一个u
    * 这个u是$\theta, \theta'$相减的结果
    * 它还是$H$的`eigen vector`
    * 它的`eigen value`$\rightarrow \lambda$ 还要小于0

实际上，`eigen value`是可以直接求出来的（上例已经求出来了），由它可以推出`eigen vector`，比如[1, 1]$^T$（自行补相关课程），往往会一对多，应该都是合理的，我们顺着共中一个u去更新$\theta$，就可以继续收敛loss。

> 实际不会真的去计算hessian matrix?

### Momentum
![](../assets/1859625-eb2390a6beff8f1d.png)

不管是较为平坦的面，还是saddle point，如果小球以图示的方式滚下去，真实的物理世界是不可能停留在那个gradient为0或接近于0的位置的，因为它有“动量”，即惯性，甚至还可能滚过local minima，这恰好是我们需要的特性。
![](../assets/1859625-5c2123a5abb30e13.png)
不但考虑当前梯度，还考虑之前累积的值（动量），这个之前，是之前所有的动量，而不是前一步的：
$
\begin{aligned}
m^0 &= 0 \\
m^1 &= -\eta g^0 \\
m^2 &= -\lambda \eta g^0 - \eta g^1 \\
&\vdots
\end{aligned}
$

### adaptive learning rate

![](../assets/1859625-3036d6027b02b243.png)

不是什么时候loss卡住了就说明到了极点(最小值，鞍点，平坦的点)

看下面这个error surface，两个参数，一个变动非常平缓，一个非常剧烈，如果应用相同的`learning rate`，要么反复横跳（过大），要么就再也挪不动步（太小）：

![](../assets/1859625-a4ad370fd272ab52.png)

### Adagrad (Root Mean Square)

于是有了下面的优化方法，思路与`l2正则化`差不多，利用不同参数本身gradient的大小来“惩罚”它起到的作用。

1. 这里用的是相除，因为我的梯度越小，步伐就可以跨得更大了。
2. 并且采用的是梯度的平方和(`Root Mean Square`)
![](../assets/1859625-ee61f47985824c5a.png)

![](../assets/1859625-2a5770ec8edaeaf6.png)

图中可以看出平缓的$\theta_1$就可以应用大的学习率，反之亦然。这个方法就是`Adagrad`的由来。不同的参数用不同的步伐来迭代，这是一种思路。

这就解决问题了吗？看下面这个新月形的error surface，不卖关子了，这个以前接触的更多，即梯度随时间的变化而不同，

![](../assets/1859625-9d0ffe171786a907.png)

### RMSProp

这个方法是找不到论文的。核心思想是在`Adagrad`做平方和的时候，给了一个$\alpha$作为当前这个梯度的权重(0,1)，而把前面产生的$\sigma$直接应用$(1-\alpha)$：

* $\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta}{\color{red}{\sigma_i^t}} g_i^t$
* $\sigma_i^t = \sqrt{\alpha(\theta_i^{t-1})^2 + (1-\alpha)(g_i^t)^2}$

![](../assets/1859625-e5ecd2f7cb3fcb27.png)

### Adam: (RMSProp + Momentum)

![](../assets/1859625-82a0f7e2d48fadb7.png)

### Learning Rate Scheduling

终于来到了最直观的lr scheduling部分，也是最容易理解的，随着时间的变化（如果你拟合有效的话），越接近local minima，lr越小。

而RMSProp一节里说的lr随时间变化并不是这一节里的随时间变化，而是设定一个权重，始终让**当前**的梯度拥有最高权重，注重的是当前与过往，而schedule则考量的是有计划的减小。

下图中，应用了adam优化后，由于长久以来横向移动累积的小梯度会突然爆发，形成了图中的局面，应用了scheduling后，人为在越靠近极值学习率越低，很明显直接就解决了这个问题。
![](../assets/1859625-61681993d7933f62.png)

而`warm up`没有在原理或直观上讲解更多，了解一下吧，实操上是很可行的，很多知名的网络都用了它：

![](../assets/1859625-59bca02513b04524.png)

要强行解释的话，就是adam的$\theta$是一个基于统计的结果，所以要在看了足够多的数据之后才有意义，因此采用了一开始小步伐再增加到大步伐这样一个过度，拿到足够的数据之后，才开始一个正常的不断减小的schedule的过程。

更多可参考：`RAdam`: https://arxiv.org/abs/1908.03265

### Summary of Optimization
![](../assets/1859625-527578692542295f.png)

回顾下`Momentum`，它就是不但考虑当前的梯度，还考虑之前所有的梯度（加起来），通过数学计算，当然是能算出它的”动量“的。

那么同样是累计过往的梯度，一个在分母（$\theta$)，一个在分子（momentum)，那不是抵消了吗？

1. momentum是相加，保留了方向
2. $\sigma$是平方和，只保留了大小


## Batch Normalization

沿着cost surface找到最低点有一个思路，就是能不能把山“铲平”？即把地貌由崎岖变得平滑点？ `batch normalization`就是其中一种把山铲平的方法。
![](../assets/1859625-2b4da7bf4fe85b71.png)

其实就是人为控制了error的范围，让它在各个feature上面的“数量级”基本一致（均值0，方差1），这样产生的error surface不会出现某参数影响相当小，某些影响又相当大，而纯粹是因为input本身量级不同的原因（比如房价动以百万计，而年份是一年一年增的）

error surface可以想象成每一个特征拥有一个轴（课程用二到三维演示），BN让每条轴上的ticks拥有差不多的度量。

然后，你把它丢到深层网络里去，你的输出的分布又是不可控的，要接下一个网络的话，你的输出又成了下一个网络的输入。虽然你在输出前nomalization过了，但是可能被极大和极小的权重w又给变了了数量级不同的输出

再然后，不像第一层，输入的数据来自于训练资料，下一层的输入是要在上一层的输出进行sigmoid之后的

再然后，你去看看sigmoid函数的形状，它在大于一定值或小于一定值之后，对x的变化是非常不敏感了，这样非常容易了出现梯度消失的现象。

于是，出于以下两个原因，我们都会考虑在输出后也接一次batch normalization::
1. 归一化（$\mu=0, \delta=1$)
2. 把输入压缩到一个（sigmoid梯度较大的）小区间内

照这个思路，我们是需要在sigmoid之前进行一次BN的，而有的教材会告诉你之前之后做都没关系，那么之后去做就丧失了以上第二条的好处。

**副作用**

* 以前$x_1 \rightarrow z_1 \rightarrow a_1$
* 现在$\tilde z_1$是用所有$z_i$算出来的，不再是独立的了

**后记1**

最后，实际还会把$\tilde z_i$再这么处理一次：
* $\hat z_i = \gamma \odot \tilde z_i + \beta$

不要担心又把量级和偏移都做回去了，会以1和0为初始值慢慢learn的。

**后记2**

推理的时候，如果batch size不够，甚至只有一条时，怎么去算$\mu, \sigma$呢？

pytorch在训练的时候会计算`moving average`of $\mu$ and $\sigma$ of the batches.(每次把当前批次的均值和历史均值来计算一个新的历史均值$\bar \mu$)
* $\bar \mu \leftarrow p \bar \mu + (1-p)\mu_t$

推理的时候用$\bar \mu, \bar \sigma$。

最后，用了BN，平滑了error surface，学习率就可以设大一点了，加速收敛。

# Classification

用数字来表示class，就会存在认为1跟2比较近与3比较远的可能（从数学运算来看也确实是的，毕竟神经网络就是不断地乘加和与真值减做对比），所以引入了one-hot，它的特征就是class之间无关联。

恰恰是这个特性，使得用one-hot来表示词向量的时候成了一个要克服的缺点。预测单词确实是一个分类问题，然后词与词之间却并不是无关的，恰恰是有距离远近的概念的，而把它还原回数字也解决不了问题，因为单个数字与前后的数字确实近了，但是空间上还是可以和很多数字接近的，所以向量还是必要的，于是又继续打补丁，才有了稠密矩阵embedding的诞生。

## softmax

softmax的一个简单的解释就是你的真值是0和1的组合(one-hot)，但你的预测值可以是任何数，因为你需要把它normalize到(0,1)的区间。

当class只有两个时，用softmax和用sigmoid是一样的。

## loss

可以继续用MeanSquare Error(MSE) $ e = \sum_i(\hat y_i - y'_i)^2$，但更常用的是：

### Cross-entropy

$e = - \sum_i \hat y_i lny'_i$

> `Minimizing cross-entropy` is equivalent to `maximizing likelihood`
![](../assets/1859625-011803fa18ea1c80.png)

linear regression是想从真值与预测值的差来入手找到最合适的参数，而logistic regression是想找到一个符合真值分布的的预测分布。

在吴恩达的课程里，这个损失函数是”找出来的“：

![](../assets/1859625-e5bfbd4de3527f60.png)

1. 首先，$\theta x$后的值可以是任意值，所以再sigmoid一下，以下记为hx
2. hx的意思就是`y为1的概率`
3. 我需要一个损失函数，希望当真值是0时，预测y为1的概率的误差应该为无穷大
    * 也就是说hx=0时，损失函数的结果应该是无穷大
    * 而hx=1时, 损失应该为0
4. 同理，当y为1时，hx=0时损失应该是无穷大，hx=1时损失为0
5. 这时候才告诉你，log函数**刚好长这样**，请回看上面的两张图

而别的地方是告诉你log是为了把概率连乘变成连加，方便计算。李宏毅这里干脆就直接告诉你公式长这样了。。。

这里绕两个弯就好了：
1. y=1时，预测y为1的概率为1， y=0时，应预测y=1的概率为0
2. 而这里是做损失函数，所以预测对了损失为0，错了损失无穷大
3. 预测为1的概率就是hx，横轴也是hx

> 课程里说softmax和cross entorpy紧密到pytorch里直接就把两者结合到一起了，应用cross entropy的时候把softmax加到了你的network的最后一层（也就是说你没必要手写）。这里说的只是pytorch是这么处理的吗？
>
> ----是的

### CE v.s. MSE

数学证明：http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Deep%20More%20(v2).ecm.mp4/index.html

![](../assets/1859625-29ba3177a9772077.png)

单看实验结果，初始位置同为loss较大的左上角，因为CE有明显的梯度，很容易找到右下角的极值，但是MSE即使loss巨大，但是却没有梯度。因此对于逻辑回归，选择交叉熵从实验来看是合理的，数学推导请看上面的链接。

# CNN

1. **Receptive field**

不管是计算机，还是人脑，去认一个物，都是去判断特定的patten（所以就会有错认的图片产生），这也说明，如果神经网络要去辨识物体，是不需要每个神经元都把整张图片看一次的，只需要关注一些特征区域就好了。（感受野, `Receptive field`)

如果你一直用3x3，会不会看不到大的patten呢？$\rightarrow$ 会也不会。

首先，小的filter当然是不可能看到它的感受野以外的部分，但是，神经网络是多层架构，你这层的输出再被卷一次，这时候每一个数字代表的就是之前的9个像素计算的结果，这一轮的9个数字就是上一层的81个像素（因为stride的原因，大部分是重复的）的计算结果，换言之，感受野大大增强了，也就是说，你只需要增加层数，就可以在小的filter上得到大的patten.

2. **filter & feature map**

从神经元角度和全连接角度出发的话，每个框其实可以有自己的参数的（即你用了64步把整个图片扫描完的话，就有64组参数），而事实上为了简化模型，可以让某些框对应同样的参数（**参数共享**），原因就是同一特征可能出现在多个位置，比如人有两只脚。

再然后，实际上每一次都是用一组参数扫完全图的，意思是在每个角落都只搜索这**一个特征**。

我们把这种机制叫`filter`，一个filter只找一种特征，乘加出来的结果叫`feature map`，即这个filter提取出来的特征图。

因此，
* 你想提取多少个特征，就得有多少个filter
* 表现出来就成了你这一层输出有多少个channel
* 这就是为什么你的图片进来是3channel，出来就是N个channel了，取决于你设计了多少个filter

3. **Pooling & subsampling**

由于图像的视觉特征，你把它放大或缩小都能被人眼认出来，因此就产生了pooling这种机制，可以降低样本的大小，这主要是为了减小运算量吧（硬件性能足够就可以不考虑它）。

4. **Data Augmentation**

CNN并不能识别缩放、旋转、裁切、翻转过的图片，因此训练数据的增强也是必要的。

## AlphaGo

**layer 1**
1. 能被影像化的问题就可以尝试CNN，围棋可以看成是一张19x19的图片
2. 每一个位置被总结出了48种可能的情况(超参1)
3. 所以输入就是19x19x48
4. 用0来padding成23x23
5. 很多patten、定式也是影像化的，可以被filter扫出来
6. 总结出5x5大小的filter就够用了（超参2）
7. 就用了192个fitler（即每一次output有48层channel)（超参3）
8. stride = 1
9. ReLU

**layer 2-12**
1. padding成 21x21
2. 192个 3x3 filter with stride = 1
3. ReLU

**layer 13**
1. 1x1 filter stride = 1
2. bias
3. softmax

其中192(个filter)这个超参对比了128，256，384等，也就是说人类并不理解它每一次都提取了什么特征。

> subsampling对围棋也有用吗？ 上面的结构看出并没有用，事实上，围棋你抽掉一行一列影响是很大的。

# Self-Attention

前面说的都是输入为一个向量（总会拉平成一维向量），如果是多个向量呢？有这样的场景吗？
* 一段文字，每一个文字都用one-hot或word-embedding来表示
    * 不但是多个向量，而且还长短不齐
* 一段语音，每25ms采样形成一个向量，步长为每10ms重复采样，形成向量序列
    * 400 sample points (16khz)
    * 39-dim MFCC
    * 80-dim filter bank output
    * 参考人类语言处理课程
* 一个Graph组向量（比如social network)
    * 每个节点（每个人的profile）就是一个向量
* 一个分子结构
    * 每个原子就是一个one-hot

**输出是什么样的？**

1. 一个向量对应一个输出
    * 文字 -> POS tagging
    * 语音 -> a, a, b, b(怎么去重也参考[人类语言处理](https://speech.ee.ntu.edu.tw/~hylee/dlhlp/2020-spring.html)课程)
    * graph -> 每个节点输出特性（比如每个人的购买决策）
2. 只有一个输出
    * 文字 -> 情绪分析，舆情分析
    * 语音 -> 判断是谁说的
    * graph -> 输出整个graph的特性，比如亲水性如何
3. 不定输出（由network自己决定）
    * 这就叫seq2seq
    * 文字 -> 翻译
    * 语音 -> 真正的语音识别

self-attention

稍稍回顾一下self attention里最重要的q, k, v的部分：

![](../assets/1859625-2e4df1ab5ca25149.png)

图示的是q2与所有的k相乘，再分别与对应的v相乘，然后相加，得到q2对应的输出：b2的过程。

下图则是矩阵化后的结论：
![](../assets/1859625-69882dd1e6b2701e.png)
具体细节看专题

真正要学的，就是图中的$W^q, W^k, W^v$

## Multi-head Self-attention
![](../assets/1859625-8ae9ebcbbd998558.png)

CNN是Self-attention的特例

![](../assets/1859625-1e5ceafb18ac5d32.png)

## Self-attention for Graph

了解更多：https://youtu.be/eybCCtNKwzA

# Transformer

Transformer是一个seq2seq的model

以下场景，不管看上去像不像是seq2seq的特征，都可以尝试用seq2seq（trnasformer）来“硬train一发”

* QA类的问题，送进去question + context，输出answer
    * 翻译，摘要，差别，情感分析，只要训练能套上上面的格式，就有可能
* 文法剖析，送入是句子，输出是树状的语法结构
    * 把树状结构摊平（其实就是多层括号）
    * 然后就用这个对应关系来当成翻译来训练（即把语法当成翻译）
* multi-label classification
    * 你不能在做multi-class classification的时候取top-k,因为有的属于一个类，有的属于三个类，k不定
    * 所以你把每个输入和N个输出也丢到seq2seq里去硬train一发，网络会自己学到每个文章属于哪“些”类别（不定个数，也像翻译一样）
* object dectection
    * 这个更匪夷所思，感兴趣看论文：https://arxiv.org/abs/2005.12872(End-to-End Object Detection with Transformers)

## Encoder

Q, K, V(relavant/similarity), zero padding mask, layer normalization, residual等, 具体看`self-attention`一节。

## Decoder

### AT v.s. NAT

我们之前用的decoder都是一个一个字地预测（输出的）
* 所以才有position-mask（用来屏蔽当前位置后面的字）

这种叫`Auto Regressive`，简称`AT`,`NAT`即`Non Auto Regressive`

![](../assets/1859625-680f5c4380c93898.png)

它一次生成输出的句子。

至于seq2seq的输出是不定长的，它是怎么在一次输出里面确定长度的，上图已经给出了几种做法：
1. 另做一个predictor来输出一个数字，表示应该输出的长度
2. 直接用一个足够长的<bos>做输入（比如300个），那输出也就有300个，取到第一个<eos>为止

因为不是一个一个生成了，好处
1. 可以平行运算。
2. 输出的长度更可控

> NAT通常表现不如AT好 (why? **Multi-mmodality**)

detail: https://youtu.be/jvyKmU4OM3c (Non-Autoregressive Sequence Generation)

### AT

在decoder里最初有让人看不懂的三个箭头从encode的输出里指出来:

![](../assets/1859625-6574d04e094f240a.png)

其实这就是`cross attention`

![](../assets/1859625-8d8049bb84081547.png)

它就是把自己第一层(self-attention后)的输出乘一个$W^q$得到的`q`，去跟encoder的输出分别乘$W^k, W^v$得到的k和v运算($\sum q \times k \times v$)得到当前位置的输出的过程。

而且研究者也尝试过各种`cross attention`的方法，而不仅仅是本文中的无论哪一层都用`encoder`最后一层的输出做q和v这一种方案：

![](../assets/1859625-12d3f05c2a546bfe.png)

## Training Tips

### 复制机制

![](../assets/1859625-46f9d4aecc642f6b.png)

一些场景，训练的时候没必要去“生成”阅读材料里提到的一些概念，只需要把它“复制”出来即可，比如上述的人名，专有名字，概念等，以及对文章做摘要等。

* Pointer Network: https://youtu.be/VdOyqNQ9aww
* Copying Mechanism in Seq2Seq https://arxiv.org/abs/1603.06393

### Guided Attention

像语音这种连续性的，需要强制指定(guide)它的attention顺序，相对而言，文字跳跃感可以更大，语音一旦不连续就失去了可听性了，一些关键字：
* Monotonic Attention
* Location-aware attention

### Beam Search

![](../assets/1859625-8b07d1dc44895d0b.png)

### Optimizing Evaluation Metrics / BLEU

* 训练的时候loss用的是cross entropy，要求loss越小越好，
* 而在evaluation的时候，我们用的是预测值与真值的`BLEU score`，要求score越大越好

* 那么越小的cross entropy loss真的能产生越高的BLEU score吗？ 未必
* 那么能不能在训练的时候也用BLEU score呢？ 不行，它太复杂没法微分，就没法bp做梯度了。

### Exposure bias

训练时候应用了`Teaching force`，用了全部或部分真值当作预测结果来训练（或防止一错到底），而eval的时候确实就是一错到底的模式了。

# Self-supervised Learning

* 芝麻街家庭：elmo, bert, erine...
* bert就是transformer的encoder

## Bert

### GLUE

GLUE: General Language Understanding Evaluation

基本上就是看以下这九个模型的得分：

![](../assets/1859625-0895ab4b9a46e931.png)

训练：
1. 预测mask掉的词(masked token prediction)
    * 为训练数据集添加部分掩码，预测可能的输出
    * 类似word2vec的C-Bow
2. 预测下一个句子（分类，比如是否相关）(next sentence prediction)
    * 在句首添加<cls>用来接分类结果
    * 用<sep>来表示句子分隔

下游任务（Downstream Task） <- Fine Tune:
1. sequence -> class: sentiment analysis
    * 这是需要有label的
    * <cls>节点对的linear部分是随机初始化
    * bert部分是pre-train的
2. sequence -> sequence(等长): POS tagging
3. 2 sequences -> class: NLI(从句子A能否推出句子B)(Natural Language Inferencee)
    * 也比如文章下面的留言的立场分析
    * 用<cls>输出分类结果，用<sep>分隔句子
4. Extraction-based Question Answering: 基于已有文本的问答系统
    * 答案一定是出现在文章里面的
    * 输入文章和问题的向量
    * 输出两个数字(start, end)，表示答案在文章中的索引

QA输出：

![](../assets/1859625-6e6c9471f5515877.png)

思路：
1. 用<cls>input<sep>document 的格式把输入摆好
2. 用pre-trained的bert模型输出同样个数的向量
3. 准备两个与bert模型等长的向量（比如768维）a, b（random initialized)
4. a与document的每个向量相乘(inner product)
5. softmax后，找到最大值，对应的位置(argmax)即为start index
6. 同样的事b再做一遍，得到end index

![](../assets/1859625-e94a647f82156c38.png)

### Bert train seq2seq

也是可能的。就是你把输入“弄坏”，比如去掉一些字词，打乱词序，倒转，替换等任意方式，让一个decoder把它还原。 -> **BART**

### 附加知识

有研究人员用bert去分类DNA，蛋白质，音乐。以DNA为例，元素为A,C,G,T,分别对应4个随机词汇，再用bert去分类（用一个英文的pre-trained model），同样的例子用在了蛋白质和音乐上，居然发现效果全部要好于“纯随机”。

如果之前的实验说明了bert看懂了我们的文章，那么这个荒诞的实验（用完全无关的随意的英文单词代替另一学科里面的类别）似乎证明了事情没有那么简单。

### More

1. https://youtu.be/1_gRK9EIQpc
2. https://youtu.be/Bywo7m6ySlk

## Multi-lingual Bert

略 

## GPT-3

训练是predict next token...so it can do generation(能做生成)

> Language Model 都能做generation
![](../assets/1859625-687d9cf507137131.png)

https://youtu.be/DOG1L9lvsDY

别的模型是pre-train后，再fine-tune， GPT-3是想实现zero-shot，

### Image

**SimCLR**

* https://arxiv.org/abs/2002.05709
* https://github.com/google-research/simclr

**BYOL**

* **B**ootstrap **y**our **o**own **l**atent
* https://arxiv.org/abs/2006.07733

### Speech

在bert上有九个任务(GLUE)来差别效果好不好，在speech领域还缺乏这样的数据库。

## Auto Encoder

也是一种`self-supervised` Learning Framework -> 也叫 pre-train, 回顾：
![](../assets/1859625-43a21995530788b6.png)

在这个之前，其实有个更古老的任务，它就是`Auto Encoder`

![](../assets/1859625-2c3764ec5ff4cb6b.png)

* 用图像为例，通过一个网络encode成一个向量后，再通过一个网络解码(reconstrucion)回这张图像（哪怕有信息缺失）
* 中间生成的那个向量可以理解为对原图进行的压缩
* 或者说一种降维

降维的课程：
* PCA: https://youtu.be/iwh5o_M4BNU
* t-SNE: https://youtu.be/GBUEjkpoxXc

有一个de-noising的Auto-encoder, 给入的是加了噪音的数据，经过encode-decode之后还原的是没有加噪音的数据

这就像加了噪音去训练bert

![](../assets/1859625-71c69ce2d3693253.png)

### Feature Disentangle

去解释auto-encoder压成的向量就叫`Feature Disentagle`，比如一段音频，哪些是内容，哪些是人物；一段文字，哪些表示语义，哪些是语法；一张图片，哪些表示物体，哪些表示纹理，等。

应用： voice conversion -> 变声器

传统的做法应该是每一个语句，都有两种语音的资料，N种语言/语音的话，就需要N份。有Feature Disentangle的话，只要有两种语音的encoder，就能知道哪些是语音特征，哪些是内容特征，拼起来，就能用A的语音去读B的内容。所以**前提**就是能分析压缩出来的向量。

### Discrete Latent Representation

如果压缩成的向量不是实数，而是一个binary或one-hot
* binary: 每一个维度几乎都有它的含义，我们只需要看它是0还是1
* one-hot: 直接变分类了。-> `unsupervised classification`

**VQVAE**

![](../assets/1859625-f8bab3e37a58d91b.png)

* Vector Quantized Variational Auot-encoder https://arxiv.org/abs/1711.00937

### Text as Representation

* https://arxiv.org/abs/1810.02851

如果压缩成的不是一个向量，而也是一段`word sequence`，那么是不是就成了`summary`的任务？ 只要encoder和decoder都是seq2seq的model

-> seq2seq2seq auto-encoder -> `unsupervised summarization`

事实上训练的时候encoder和decoder可能产生强关联，这个时候就引入一个额外的`discriminator`来作判别:
![](../assets/1859625-03f7f375744bc8cf.png)

有点像cycle GAN，一个generator接一个discriminator，再接另一个generator

### abnormal detection

![](../assets/1859625-39b0acc3bf74a2ef.png)

* Part 1: https://youtu.be/gDp2LXGnVLQ
* Part 2: https://youtu.be/cYrNjLxkoXs
* Part 3: https://youtu.be/ueDlm2FkCnw
* Part 4: https://youtu.be/XwkHOUPbc0Q
* Part 5: https://youtu.be/Fh1xFBktRLQ
* Part 6: https://youtu.be/LmFWzmn2rFY
* Part 7: https://youtu.be/6W8FqUGYyDo

# Adversarial Attack

给你一张猫的图片，里面加入少许噪音，以保证肉眼看不出来有噪音的存在：
1. 期望分类器认为它不是猫
2. 期望分类器认为它是一条鱼，一个键盘...

比如你想要欺骗垃圾邮件过滤器

* 找到一个与$x^0$非常近的向量x
* 网络正常输出y
* 真值为$\hat y$
* $L(x) = -e(y, \hat y)$
* $x^* = arg\underset{d(x^0, x) \leq \epsilon}{\rm min}\ L(x)$ 即要找到令损失最大的x
    1. 这里L(x)我们取了反
    2. $\epsilon$越小越好，指的是$x^0$要与x越接近越好（欺骗人眼）
* 如果还期望它认成是$y^{target}$，那就再加上与其的的损失
* $L(x) = -e(y, \hat y) + e(y, y^{target})$
* 注意两个error是反的，一个要求越远越好(真值），一个要求越近越好（target)

怎么计算$d(x^0, x) \leq \epsilon$呢？

![](../assets/1859625-683386630a1ecc47.png)

图上可知，如果都改变一点点，和某一个区域改动相当大，可能在L2-norm的方式计算出来是一样的，但是在L-infinity看来是不一样的（它只关心最大的变动）。

显然L-infinity更适合人眼的逻辑，全部一起微调人眼不能察觉，单单某一块大调，人眼是肯定可以看出来的。

而如果是语音的话，可能耳朵对突然某个声音的变化反而不敏感，整体语音风格变了却能立刻认出说话的人声音变了，这就要改变方案了。

## Attack Approach

如何得到这个x呢？其实就是上面的损失函数。以前我们是为了train权重，现在train的就是x本身了。
1. 损失达到我们的要求 （有可能这时候与原x相关很远）
2. 与原x的距离达到我们的要求, 怎么做？
    * 其实就是以$x^0$为中心，边长为$2\epsilon$的矩形才是期望区域
    * 如果update后，$x^t$仍然落在矩形外，那么就在矩形里找一个离它最近的点，当作本轮更新后的$x^t$，进入下一轮迭代

Fast Gradient Sign Method(FGSM): https://arxiv.org/abs/1412.6572
* 相比上面的迭代方法，FGSM只做一次更新
* 就是根据梯度，判断是正还是负，然后把原x进行一次加减$\epsilon$的操作（其实等于是落在了矩形的四个点上）
* 也就是说它直接取了四个点之一作为$x^0$

## White Box v.s. Black Box

讲上述方法的时候肯定都在疑惑，分类器是别人的，我怎么可能拿到别人的模型来训练我的攻击器？ -> **White Box Attack**

那么`Black Box Attack`是怎么实现的呢？
1. 如果我们知道对方的模型是用什么数据训练的话，我们也可以训练一个类似的(proxy network)
    * 很大概率都是用公开数据集训练的
2. 如果不知道的话呢？就只能尝试地丢一些数据进去，观察（记录）它的输出，然后再用这些测试的输入输出来训练自己的proxy network了。

* one pixel attack
    * https://arxiv.org/abs/1710.08864
    * https://youtu.be/tfpKIZIWidA
* universal adversarial attack
    * 万能noise
    * https://arxiv.org/abs/1610.08401
* 声音
* 文本
* 物理世界
    * 比如欺骗人脸识别系统，去认成另一个人
    * 又比如道路环境，车牌识别等，也可以被攻击
    * 要考虑摄像头能识别的分辨率
    * 要考虑训练时候用的图片颜色与真实世界颜色不一致的问题
* Adversarial Reprogramming
* Backdoor in Model
    * attack happens at the training phase
    * https://arxiv.org/abs/1804.00792
    * be careful of unknown dataset...

## Defence

### Passive Defense（被动防御）

进入network前加一层filter
* 稍微模糊化一点，就去除掉精心设计的noise了
    * 但是同时也影响了正常的图像
* 对原图进行压缩
* 把输入用Generator重新生成一遍

如果攻击都知道你怎么做了，其实很好破解，就把你的filter当作network的一部分重新开始设计noise，所以可以选择加入随机选择的一些预处理(让攻击者不可能针对性地训练)：

![](../assets/1859625-a467e272d69cc306.png)

### Proactive Defense（主动防御）

训练的时候就训练比较不容易被攻破的模型。比如训练过程中加入noise，把生成的结果重新标注回真值。

* training model
* find the problem
* fix it

有点类似于`Data Augmentation`

仍然阻挡不了新的攻击算法，即你对数据进行augment之外的范围。

# Explainable Machine Learning(可解释性)

* correct answers $\neq$ intelligent
* 很多行业会要求结果必须可解释
    * 银行，医药，法律，驾驶....

**Local Explanation**

Why do you thing **this image** is a cat?

**Global Explanation**

What does a "**cat**" look like?

1. 遮挡或改变输入的某些部分，观察对已知输出的影响
    * （比如拦到某些部分确实认不出图像是一条狗了）
2. 遮挡或改变输入的某些部分，把两种输出做loss，对比输入变化与loss变化：
    * $|\frac{\varDelta e}{\varDelta x}| \rightarrow \frac{\partial e}{\partial x_n}$

把上述（任一种）每个部分（像素，单词）的影响结果输出，就是：`Saliency Map`

## Saliency Map
![](../assets/1859625-a69f229d00a9d750.png)

图1，2就是为了分辨宝可梦和数码宝贝，人类一般很难区分出来，但机器居然轻松达到了98%的准确率，经过绘制`Saliency Map`，发现居然就是图片素材（格式）的原因，一个是png，一个是jpg，造成背景一个是透明一个是不透明的。

也就是说，能发现机器判断的依据不是我们关注的本体（高亮部分就是影响最大的部分，期望是在动物身上）

第三张图更可笑，机器是如何判断这是一只马的？居然也不是马的本体，而是左下角，标识图片出处的文字，可能是训练过程中同样的logo过多，造成了这个“人为特征”。

解决方案：

### Smooth Gradient

随机给输入图片加入噪点，得到saliency map（们），然后取平均

![](../assets/1859625-e3f0f1a301cbd178.png)

### Integrated gradient(IG)

一个特征在从无到有的阶段，梯度还是明显的，但是到了一定程度，特征再增强，对gradient影响也不大了，比如从片子来判断大象，到了一定长度，一张图也不会“更像大象”

一种思路：https://arxiv.org/abs/1611.02639

## global explaination

**What does a filter detect?**

如果经过某层（训练好的）filter，得到的feature map一些位置的值特别大，那说明这个filter提取的就是这类特征/patten。

我们去"创造"一张包含了这种patten的图片：$X^* = arg\ \underset{X}{\rm max}\sum_i\sum_j a_{ij}$，即这个图片是“训练/learn“出来的，通过找让X的每个元素($a_{ij}$)在被filter乘加后结果最大的方式。 -> `gradient ascent`

然后再去观察$X^*$有什么特征，就基本上可以认定这个（训练好的）filter提取的是什么样的patten了。
![](../assets/1859625-be596d0584d49d4e.png)

> `adversarial attack` 类似的原理，但这是对单filter而言。如果你想用同样的思路去让输出y越大越好，得到X，看X是什么，得到的X大概率都是一堆噪音。如果能生成图像，那是`GAN`的范畴了。

于是，尝试再加一个限制，即不但要让y最大，还要让X看起来最有可能像一个数字：

* $R(X)$: how likely X is a digit 
* $X^* = arg\ \underset{X}{\rm max}y_i + \color{red}{R(X)}$
* $R(X) = -\sum_{i,j}|X_{i,j}|$ 比如这个规则，期望每个像素越黑越好

# Domain Adaptation

`Transfer Learning`的一种，在训练数据集和实际使用的数据集不一样的时候。 https://youtu.be/qD6iD4TFsdQ

需要你对`target domain`的数据集有一定的了解。

有一种比较好的情况就是，target domain既有数据，还有标注（但不是太多，如果太多的话就不需要`source domain`了，直接用target来训练就好了），那就像bert一样，去`fine tune`结果，要注意的是标本量过小，可能很容易`overfitting`.

如果target doamin有**大量**资料，但是没有标注呢？

## Domain Adversarial Training
![](../assets/1859625-dab4c03bac102639.png)

* 把source domain的network分为特征提取器（取多少层cnn可以视为超参，并不一定要取所有层cnn）和分类器
* 然后在特征取层之后跟另一个分类器，用来判断图像来自于source还是target（有点像`Discriminator`
* 与真值有一个loss，source, target之间也有一个loss，要求找到这样的参数组分别让两个loss最小
* loos和也应该最小（图中用的是减，但其实$L_d$的期望是趋近于0，不管是正还是负都是期望越小越好）（不如加个绝对值？）
* 每一小块都有一组参数，是一起训练的
* 目的就是既要逼近训练集的真值，还要训练出一个网络能模糊掉source和target数据集的差别

### Limit

![](../assets/1859625-6891c5cbf5d2c728.png)

如果target数据集如上图左，显然结果是会比上图右要差一点的，也就是说尽量要保持同分布。在这里用了另一个角度，就是让数据**离boundary越远越好**

* Decision-boundary Iterative Refinement Training with a Teacher(`DIRT-T`)
    * https://arxiv.org/abs/1802.08735
* Maximum Classifier Discrepancy https://arxiv.org/abs/1712.02560

## More

* 如果source 和 target 里的类别不完全一样呢？
    * Universal domain adaptation
* 如果target既没有label，数据量也非常少（比如就一张）呢？
    * Test Time Training(TTT) https://arxiv.org/abs/1909.13231

**Domain Generalization**
![](../assets/1859625-0c290ef2c9a19b50.png)

# Deep Reinforcement Learning (RL)

* **Environment** 给你 `Observation`
* **Actor** 接收入 `Observation`, 输出 `Action`
* `Action` 反馈给 **Environment**, 计算出 `Reward` 反馈给 **Actor**
* 要求 `Reward` 最大

与 GAN 的不同之处，不管是生成器还是判别器，都是一个network，而RL里面，Actor和Reward都是黑盒子，你只能看到结果。

## Policy Gradient

https://youtu.be/W8XF3ME8G2I

1. 先是用很类似监督学习的思路，给每一步的最优（或最差）方案一个label，有label就能做loss。先把它变成一个二分类的问题。
2. 打分还可以不仅仅是“好”或“不好”，还可以是一个程度，比如1.5比0.5的“支持”力度要大一些，而-10显然意味着你千万不要这么做，非常拒绝。
3. 比如某一步，可以有三种走法，可以用onehot来表示，其中一种走法可以是[1,0,0]$^T$，表示期望的走法是第一种。
4. 但是也可以是[-1,0,0]$^T$，标识这种走法是不建议的
5. 也可以是[3.5,0,0]$^T$等
6. 后面会用`1, -1, 10, 3.5`这样的scalar来表示，但要记住其实它们是ont-hot中的那个非零数。

现实世界中很多场景不可能执行完一步后就获得reward，或者是全局最佳的reward（比如下围棋）。

**v1**

一种思路是，每一步之后，把游戏/棋局进行完，把当前reward和后续所有步骤的reward加一起做reward -> `cumulated reward` $\rightarrow G_t = \sum_{n=t}^Nr_n$

**v2**

这种思路仍然有问题，游戏步骤越长，当前步对最终步的影响越小。因此引入一个小于1的权重$\gamma < 1$: $G_1' = r_1 + \gamma r_2 + \gamma^2r_3 + \cdots$

这样越远的权重越小： $G_t' = \sum_{n=t}^N \color{red}{\gamma^{n-t}} r_n$

> 注意，目前得到的`G`就是为了给每一次对observation进行的action做loss的对象。

**v3**

标准化reward。你有10分，是高是低？如果所有人都是20分，那就是低分，所以与G做对比的时候，通常要减去一个合适的值`b`，让得分的分布有正有负。

**Policy Gradient**

普通的gradient descent是搜集一遍数据，就可以跑for循环了，而PG不行，你每次得到梯度后，要重采一遍样，其实也很好理解，你下了某一步，经过后续50步后，输了，你的下一轮测试应该是下一盘随机的棋，而不是把更新好的参数再用到同一盘棋去。

还是不怎么好理解，至少要知道，我做参数是不为了训练出这一盘棋是怎么下出来的，而是根据这个（大多是输了的）结果，以及学到的梯度，去下一盘新的棋试试。

## Actor Critic

**Critic**: 
* Given `actor` $\theta$, how good it is when `observing` s (and taking action a)

**Value function** $V^\theta(s)$:
* 使用actor $\theta$的时候，预测会得到多少的`cumulated reward`
* 分高分低其实还是取决于actor，同样的局面，不同的actor肯定拿的分不同。

### Monte-Carlo based approach (MC)

蒙特卡洛搜索，正常把游戏玩完，得到相应的G.

### Temporal-difference approach (TD)

不用玩完整个游戏，就用前后时间段的数据来得到输出。
![](../assets/1859625-eae0e77ede8b2ae9.png)

关键词：
* 我们既不知道v(t+1)，也不知道v(t)，但确实能知道`v(t+1)-v(t)`.

![](../assets/1859625-437a2be80e108b60.png)

这个例子没看懂，后面七次游戏为什么没有sa了？

**v3.5**

上文提到的V可以用来作更早提到的b:
* $\{S_t, a_t\}\ A_t = G_t' - V^\theta(S_t)$
* 回顾一下，$V^\theta(S_t)$是看到某个游戏画面时算出来的reward
* 它包含$S_t$状态下，后续各种步骤的reward的平均值
* 而$G_t'$则是这一步下的rewared
* 两个数相减其实就是看你的这一步是比平均水平好还是差
* 比如你得到了个负值，代表在当前场景下，这个actor执行的步骤是低于平均胜率的，需要换一种走法。

**v4**

3.5版下，G只有一个样本（一次游戏）的结果，这个版本里，把st再走一步，试难$S_{t+1}$的各种走法下reward的平均值，用它来替换G'，而它的值，就是当前的reward加上t+1时刻的V:
* $r_t + V^\theta(S_{t+1}) - V^\theta(S_t)$

这就是：

### Advantage Actor-Critic
![](../assets/1859625-07e8e3fb880671ca.png)

就看图而言，感觉就是坚持这一步走完，后续所有可能的rewawrd， 减去， 从这一步开始就试验所有走法的reward

![](../assets/1859625-2ff6bf8381c80d55.png)

More:

Deep Q Network (DQN)
* https://arxiv.org/abs/1710.02298
* https://youtu.be/o_g9JUMw1Oc
* https://youtu.be/2-zGCx4iv_k

## Reward Shaping

前面说过很多场景要得到reward非常困难（时间长，步骤长，或根本不会结束），这样的情况叫`sparse reward`，人类可以利用一些已知知识去人为设置一些reward以增强或削弱机器的某些行为。

比如游戏：
1. 原地不动一直慢慢减分
2. 每多活一秒也慢慢减分（迫使你去获得更高的reward, 避免学到根本就不去战斗的方式）
3. 每掉一次血也减分
4. 每杀一个敌人就加分
5. 以此类推，这样就不至于要等到一场比赛结束才有“一个”reward

又比如训练机械手把一块有洞的木板套到一根棍子上：
1. 离棍子越近，就有一定的加分
2. 其它有助于套进去的规则

还可以给机器加上**好奇心**，让机器看到有用的“新的东西”也加分。

## No Reward, learn from demostration

只有游戏场景才会有明确的reward，大多数现实场景都是没有reward的，比如训练自动驾驶的车，或者太过死板的reward既不能适应变化，也容易被打出漏洞，比如机器人三定律里，机器人不能伤害人类，却没有禁止囚禁人类，又比如摆放盘子，却没有给出力度，等盘子摔碎了，再去补一条𢱨碎盘子就负reward的规则，也晚了，由此引入模仿学习：

### Imitation Learning

略

# Life-Long Learning

持续学习，机器学习到一个模型后，继续学下一个模型（任务）。

1. 为什么不一个任务学一个模型
    * 不可能去存储所有的模型
    * 一个任务的知识不能转移到另一个任务
2. 为什么不直接用迁移学习（迁移学习只关注迁移后的新任务）

## Research Directions

### Selective Synaptic Plasticity

选择性的神经突触的可塑性？（Regularization-based Approach）

**Catastrophic Forgetting** 灾难性的遗忘
![](../assets/1859625-135d5398dcfb50bf.png)

在任务1上学到的参数，到任务2里接着训练，顺着梯度到了任务2的最优参数，显然不再是任务1的做以参，这叫灾难性的遗忘

一种思路： 

任务2里梯度要更新未必要往中心，也可以往中下方，这样既在任务2的低loss区域，也没有跑出任务1的低loss区域，实现的方式是找到对之前任务影响比较小的参数，主要去更新那些参数。比如上图中，显然$\theta_1$对任务1的loss影响越小，但是更新它之后会显著影响任务2的loss，而$\theta_2$的改动才是造成任务1loss变大的元凶。

![](../assets/1859625-5f15705b33b403c6.png)

Elastic Weight Consolidation(EWC)
* https://arxiv.org/abs/1612.00796

Synaptic Intelligence(SI)
* https://arxiv.org/abs/1703.04200

Memory Aware Synapses(MAS)
* https://arxiv.org/abs/1711.09601

RWalk
* https://arxiv.org/abs/1801.10112

Sliced Cramer Preservation(SCP)
* https://openreview.net/forum?id=BJge3TNKwH
![](../assets/1859625-338099d4d7776dd0.png)

### Memory Reply

1. 在训练task1的时候，同时训练一个相应的generator
2. 训练task2的时候，用task1的generator生成pseudo-data，一起来训练生成新的model
3. 同时也训练出一个task1&2的generator
4. ...

# Network Compress

## pruning (剪枝)

Networks ar typically over-parameterized (there is significant redundant weights or neurons)

* 可以看哪些参数通常比较大，或值的变化不影响loss（梯度小）-> 权重，为0的次数少 -> 神经元 等等
* 剪枝后精度肯定是会下降的
* 需要接着fine-tune
* 一次不要prune to much
* 剪参数和剪神经元效果是不一样的
    * 剪参数会影响矩阵运算，继而影响GPU加速

那么为什么不直接train一个小的network呢？
* 小的network通常很难train到同样的准确率。 （大乐透假说）

## Knowledge Distillation (知识蒸馏)

老师模型训练出来的结果，用学生模型（小模型）去模拟（即是模拟整个输出，而不是模拟分类结果），让小模型能达到大模型同样的结果。

一般还会在输出的softmax里面加上温度参数（即平滑输出，不同大小的数除一个大于1的数，显然越大被缩小的倍数也越大，比如100/10=10，少了90，10/10=1, 只少了9，差别也从90变成了9）(或者兴趣个极端的例子，T取无穷大，那么每个输出就基本相等了)

## Parameter Quantization

1. Using less bits to represent a value
2. Weight clustering
    * 把weights分成预先确定好的簇（或根据分布来确定）
    * 对每簇取均值，用均值代替整个簇里所有的值
3. represent frequent clusters by less bits, represent rare clusters by more bits
    * Huffman encoding

极限，`Binary Weights`，用两个bits来描述整个网络，扩展阅读。

## Depthwise Separable Convolution

回顾下CNN的机制，参数量是：
* 卷积核的大小 x 输入图像的通道数 x 输出的通道数
* ($k\times k$) x in_channel x out_channel
![](../assets/1859625-c15dee267b4ec8d1.png)

而`Depthwise Separable Convolution`由两个卷积组成：
1. Depthwise Convolution
    * 很多人对CNN的误解刚好就是Depthwise Convolution的样子，即一个卷积核对应一个输入的channel（事实上是一组卷积核对应所有的输入channel）
    * 因此它的参数个数 k x k x in_channel
2. PointWise Convolution
    * 这里是为了补上通道与通道这间的关系
    * 于是用了一个1x1的`标准`卷积（即每一组卷积核对应输入的所有通道）
    * 输出channel也由这次卷积决定
    * 应用标准卷积参数量：(1x1) x in_channel x out_channel

![](../assets/1859625-6043951a828b505c.png)

两个参数量做对比, 设`in_channel = I`, `out_channel = O`
1. $p_1 = (k\times k) \times I \times O$
2. $p_2 = (k\times k) \times I + (1\times 1) \times I \times O = (k\times k) \times I + I \times O$
3. $\frac{p_2}{p_1} = \frac{I\cdot(k^2 + O)}{I\cdot{k^2\cdot O}}
= \frac{1}{O} + \frac{1}{k^2} \approx \frac{1}{k^2} 
$

O代表out_channel，大型网络里256，512比比皆是，所以它可以忽略，那么前后参数量就由$k^2$决定了，如果是大小为3的卷积核，参数量就变成1/9了，已经是压缩得很可观了。

### Low rank approximation

上面是应用，原理就是`Low rank approximation`

以全连接网络举例
1. 如果一个一层的网络，输入`N`， 输出`M`，参数为`W`，那么参数量是`MxN`
2. 中间插入一个线性层`K`，
    * 参数变成：`V`:N->K, `U`:K->M,
    * 参数量：`NxK` + `KxM`
3. 只要K远小于M和N（比如数量级都不一致），那么参数量是比直接MxN要小很多的
4. 这也限制了能够学习的参数的可能性（毕竟原始参数量怎么取都行）
    * 所以叫`Low rank` approximation

**to learn more**

SqueezeNet
* https://arxiv.org/abs/1602.07360

MobileNet
* https://arxiv.org/abs/1704.04861

ShuffleNet
* https://arxiv.org/abs/1707.01083

Xception
* https://arxiv.org/abs/1610.02357

GhostNet
* https://arxiv.org/abs/1911.11907

## Dynamic Computation

1. 同一个网络，自己来决定计算量，比如是在不同的设备上，又或者是在同设备的不同时期（比如闲时和忙时，比如电量充足和虚电时）
2. 为什么不为不同的场景准备不同的model呢？
    * 反而需要更大的存储空间，与问题起源（资源瓶颈）冲突了。

### Dynamic Depth

在部分layer之后，每一层都插一个额外的layer，提前做预测和输出，由调用者根据具体情况决定需要多深的depth来产生输出。

训练的时候既要考虑网络终点的loss，还要考虑所有提前结束的layer的softmax结果，加到一起算个大的Loss

Multi-Scale Dense Network(MSDNet)
* https://arxviv.org/abs/1703.09844

### Dynamic Width

训练的时候（同时？）对不同宽度（即神经元个数，或filter个数）进行计算（全部深度），也是把每种宽度最后产生的loss加起来当作总的Loss

在保留的宽度里，参数是一样的（所以应该就是同一轮训练里的参数了）

Slimmable Neural Networks
* https://arxiv.org/abs/1812.08928

### Computation based on Sample Difficulty

上述决定采用什么样的network/model的是人工决定的，那么有没有让机器自己决定采用什么网络的呢？

比如一张简单的图片，几层或一层网张就能得到结果，而另一张可能前景和或背景更复杂的图片，需要很多层才能最终把特征提取出来，应用同一个模型的话就有点资源浪费了。

* SkipNet: Learning Dynamic Routing in Convolutional Networks 
* Runtime Neural Pruning
* BlockDrop: Dynamic Inference Paths in Residual Networks

# Meta Learning

* 学习的学习。
* 之前的machine learning，输出是明确的任务，比如是一个数字，还是一个分类；而meta-learning，输出是一个model/network，用这个model，可以去做machine learning的任务。
* 它就相当于一个“返函数的函数”
* meta-learning 就是让机器学会去架构一个网络，初始化，学习率等等 $\leftarrow \varPhi$: `learnable components`
    * categorize meta learning based on what is learnable

> 不再深入

