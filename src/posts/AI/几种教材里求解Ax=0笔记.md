---
layout: post
title: 几种教材里求解Ax=0笔记
slug: 几种教材里求解Ax=0笔记
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - AI
---

如果一个矩阵化简为
$$
A=
\left[
    \begin{array}{cccc|c}
    1&2&2&2&0 \\
    0&0&1&2&0 \\
    0&0&0&0&0
    \end{array}
\right] \tag{0}
$$

求解$\bf{A}\it\vec{x}=0$

对比在不同教材中的解题思路。

## 可汗学院解法

先继续化简为`Reduced Row Echelon Form` (RREF)
$$
\left[
    \begin{array}{cccc|c}
    1&2&0&-2&0 \\
    0&0&1&2&0 \\
    0&0&0&0&0
    \end{array}
\right] \tag{1. 1}
$$

还原为方程组:
$$ 
\begin{cases}
    x_1=-2x_2+2x_4 \\
    x_3=-2x_4\\
\end{cases} \tag{1.2}
$$

用$x_2$和$x_4$来表示$x_1$和$x_3$，填满矩阵相应位置即可得解：
$$
\left[\begin{smallmatrix} x_1\\x_2\\x_3\\x_4 \end{smallmatrix}\right]=
x_2 \left[\begin{smallmatrix} -2\\1\\0\\0 \end{smallmatrix}\right] +
x_4 \left[\begin{smallmatrix} 2\\0\\-2\\1 \end{smallmatrix}\right] \tag{1.3}
$$
如果不是太直观的话，其实就是把以下方程写成了矩阵的形式：
$$
\begin{cases}
    x_1=-2x_2+2x_4 \\
    x_2=x_2\\
    x_3=-2x_4\\
    x_4=x_4
\end{cases}\tag{1. 4}
$$

----

## 剑桥教材解法

> 《Mathematics for Machine Learning》
by Marc Peter Deisenroth, A Aldo Faisal, Cheng Soon Ong,
Cambridge University

化简为`RREF`后，观察到$c_1$和$c_3$列可组成一个单位矩阵（`identity matrix`）$\left[\begin{smallmatrix} 1&0\\0&1 \end{smallmatrix}\right]$

> 如果是解$\bf{A}\it\vec{x}=b$，此时可用此矩阵求出特解，但此处是0，所以此步省略，直接求通解

我们用$c_1$和$c_3$来表示其它列：
$$
\begin{cases}
c_2=2c_1 \\
c_4=-2c_1+2c_3
\end{cases} \tag{2.1}
$$
我们利用$c_2-c_2=0, c_4-c_4=0$来构造0值（通解都是求0）：
$$
\begin{cases}
2c_1-\color{green}{c_2}=0 \\
-2c_1+2c_3-\color{green}{c_4}=0
\end{cases} \tag{2.2}
$$
补齐方程，整理顺序（以便直观地看到系数）得：
$$
\begin{cases}
\color{red}2c_1\color{red}{-1}c_2+\color{red}{0}c_3+\color{red}{0}c_4=0 \\
\color{red}{-2}c_1+\color{red}0c_2+\color{red}2c_3\color{red}{-1}c_4=0
\end{cases} \tag{2. 3}
$$

因为矩阵乘向量可以理解为矩阵和`列向量`$\vec{c}$与向量$x$的点积之和$\sum_{i=1}^4 x_ic_i$，所以红色的系数部分其实就是$(x_1, x_2, x_3, x_4)$，得解：
$$
\left\{x\in\mathbb{R}^4:x=\lambda_1\left[\begin{smallmatrix} 2\\-1\\0\\0 \end{smallmatrix}\right]+\lambda_2\left[\begin{smallmatrix} 2\\0\\-2\\1 \end{smallmatrix}\right],\lambda_1,\lambda_2\in\mathbb{R}\right\} \tag{2.4}
$$

> 与**可汗学院**的解得到的两个向量比较下，是一样的，都是$[2,-1,0,0]^T$和$[2,0,-2,1]^T$。

----

##麻省理工教材解法

> 《Introduction to Linear Alegebra》
by Gilbert Strang, 
Massachusetts Institute of Technology

无需继续化简为`RREF`，直接对方程组：
$$ 
\begin{cases}
    x_1=-2x_2+2x_4 \\
    x_3=-2x_4\\
\end{cases} \tag{3.1}
$$
使用特解。考虑到$x_1,x_3$为主元（`pivot`），那么分别设$[\begin{smallmatrix} x_2 \\ x_4 \end{smallmatrix}]$ 为$[\begin{smallmatrix} 1 \\ 0 \end{smallmatrix}]$ 和$[\begin{smallmatrix} 0 \\ 1 \end{smallmatrix}]$ 。
两种情况各代入一次，解出$x_1,x_3$，仍然是$[2,\color{red}{-1},0,\color{red}0]^T$和$[2,\color{red}0,-2,\color{red}1]^T$，红色标识了代入值，黑色即为代入后的解。

`MIT`不止提供了这一个思路，解法二如下：

这次需要化简为`RREF`，然后互换第`2`列和第`3`列（**`记住这次互换`**），还记得剑桥的方法里发现$c_1,c_3$能组成一个单位矩阵吗？这里的目的是通过移动列，直接在表现形式上变成单位矩阵：
$$
\left[
    \begin{array}{cc:cc}
    1&0&2&-2\\
    0&1&0&2\\
    \hdashline
    0&0&0&0
    \end{array}
\right] \tag{3.2}
$$
这里把用虚线反矩阵划成了四个区，左上角为一个`Identity Matrix`，我们记为`I`，右上角为自由列，我们记为`F`，矩阵（这次我们标记为**R**）变成了
$$
\bf{\it{R}}=
\begin{bmatrix}
I&F\\
0&0
\end{bmatrix} \tag{3. 3}
$$
求解$\bf{\it{R}}\it\vec{x}=0$，得到$x=\left[\begin{smallmatrix} -F\\I \end{smallmatrix}\right]$，把**F**和**I**分别展开(`记得F要乘上-1`)：
$$
\begin{bmatrix}
-2&2\\
0&-2\\
1&0\\
0&1
\end{bmatrix} \tag{3.4}
$$
还记得前面加粗提示的交换了两列吗？我们交换了两列，倒置后，我们要把第`2, 3`**行**给交换一下：
$$
\begin{bmatrix}
-2&2\\
1&0\\
0&-2\\
0&1
\end{bmatrix} \tag{3.5}
$$

是不是又得到了两个熟悉的$[2,-1,0,0]^T$和$[2,0,-2,1]^T$。？

>当时看到Gilbert教授简单粗暴地用$[\begin{smallmatrix} 1 \\ 0 \end{smallmatrix}]$ 和$[\begin{smallmatrix} 0 \\ 1 \end{smallmatrix}]$ 直接代入求出解，道理都不跟你讲，然后又给你画大饼，又是F又是I的，觉得可能他的课程不适合初学者，LOL。不过，这些Gilbert教授在此演示的解法并不适用于$\bf{A}\it\vec{x}=b$。

在此特用笔记把几本教材里的思路都记录一下。
