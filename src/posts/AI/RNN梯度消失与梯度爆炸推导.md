---
layout: post
title: RNN梯度消失与梯度爆炸推导
slug: RNN梯度消失与梯度爆炸推导
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - AI
---

![](../assets/1859625-1a8b2a5bad603588.png)
$
\large
\begin{aligned}
h_t &=\sigma(z_t) = \sigma(Ux_t+Wh_{t-1} + b) \\
y_t &= \sigma(Vh_t + c)
\end{aligned}
$

## 梯度消失与爆炸

假设一个只有 3 个输入数据的序列，此时我们的隐藏层 h1、h2、h3 和输出 y1、y2、y3 的计算公式：

$
\large
\begin{aligned}
h_1 &= \sigma(Ux_1 + Wh_0 + b) \\
h_2 &= \sigma(Ux_2 + Wh_1 + b) \\
h_3 &= \sigma(Ux_3 + Wh_2 + b) \\
y_1 &= \sigma(Vh_1 + c) \\
y_2 &= \sigma(Vh_2 + c) \\
y_3 &= \sigma(Vh_3 + c)
\end{aligned}
$

RNN 在时刻 t 的损失函数为 Lt，总的损失函数为 $L = L1 + L2 + L3 \Longrightarrow  \sum_{t=1}^TL_T$

t = 3 时刻的损失函数 L3 对于网络参数 U、W、V 的梯度如下：

$$
\begin{aligned}
\frac{\partial L_3}{\partial V} &= \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial V} \\
\frac{\partial L_3}{\partial U} &= \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial h_3} \frac{\partial h_3}{\partial U} + \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial h_3} \frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial U} + \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial h_3} \frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial h_1} \frac{\partial h_1}{\partial U} \\
\frac{\partial L_3}{\partial W} &= \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial h_3} \frac{\partial h_3}{\partial W} 
\+ \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial h_3} \frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial W} 
\+ \frac{\partial L_3}{\partial y_3} \frac{\partial y_3}{\partial h_3} \frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial h_1} \frac{\partial h_1}{\partial W} \\
\end{aligned}
$$

其实主要就是因为：

* 对V求偏导时，$h_3$是常数
* 对U求偏导时：
    * $h_3$里有U，所以要继续对h3应用`chain rule`
    * $h_3$里的$W, b$是常数，但是$h_2$里又有U，继续`chain rule`
    * 以此类推，直到$h_0$
* 对W求偏导时一样

所以：

1. 参数矩阵 V (对应输出 $y_t$) 的梯度很显然并没有长期依赖
2. U和V显然就是连乘($\prod$)后累加($\sum$) 

$$
\begin{aligned}
\frac{\partial L_t}{\partial U} = \sum_{k=0}^{t} \frac{\partial L_t}{\partial y_t} \frac{\partial y_t}{\partial h_t}
(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}})
\frac{\partial h_k}{\partial U} \\
\frac{\partial L_t}{\partial W} = \sum_{k=0}^{t} \frac{\partial L_t}{\partial y_t} \frac{\partial y_t}{\partial h_t}
(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}})
\frac{\partial h_k}{\partial W}
\end{aligned}
$$

其中的连乘项就是导致 RNN 出现梯度消失与梯度爆炸的罪魁祸首，连乘项可以如下变换：

* $h_j = tanh(Ux_j + Wh_{j-1} + b)$
* $\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}} =\prod_{j=k+1}^{t} tanh' \times W$

tanh' 表示 tanh 的导数，可以看到 RNN 求梯度的时候，实际上用到了 (tanh' × W) 的连乘。当 (tanh' × W) > 1 时，多次连乘容易导致梯度爆炸；当 (tanh' × W) < 1 时，多次连乘容易导致梯度消失。
