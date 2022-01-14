---
layout: post
title: 一张图说清匈牙利算法（Hungarian-Algorithm）
slug: 一张图说清匈牙利算法（Hungarian-Algorithm）
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - AI
---

做多目标跟踪的时候会碰到这个算法，每个人都有自己的说法讲清楚这个算法是干什么的？我的老师就跟我说过是什么给工人分配活干（即理解为`指派问题`），网上还看到有说红娘尽可能匹配多的情侣等，透过这些感性理解，基本上就能理解大概是最大匹配的问题了。

然后加了限制：后来者优先。即后匹配的**能**抢掉前人已匹配的对象，这个是有数学依据还是只是一种实现思路我就没深究了。

我的理解不会比别人更高级，之所以能用一张图说清楚，只不过是我作图的时候发现可以把过程画在一张图里，只需要把图示标清楚就好了，这样就不需要每一步画一张图了，一旦理解了，哪怕忘了，一瞅这张图也能立刻回忆起来。

先上数据：
```python
import numpy as np

relationship_matrix = np.array([
    [1,1,0,1,0,0,0],
    [0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1],
    [0,0,1,0,0,1,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0]
], dtype=bool)
```
你可以理解为6个工人，7个工作，6个男孩，7个女孩等，当然，6行7列，这么直观理解也是一点问题都没有的。

算法匹配过程如下：
![](../assets/1859625-288e5208f101f44f.png)
* 灰蓝线就是被抢掉的
* 绿线就是抢夺失败的
* `紫线`是被抢了后找候选成功的
* `红线`是一次性成功的

其中被抢的和抢夺失败的还加了删除线，这是为了强调。匹配成功的就是`红线`和`紫线`，也就是说，我们匹配出来的是：
```python
[0,1], [1,4], [2,0], [3,2], [4,3]
```

甚至可以这么表示这个过程：
```
x0,y0
x1,y1
x2,y0 -> x0,y1 -> x1->y4 (x2抢x0的,x0抢x1的)
x3,y2
x4,y3
x5,y3 -> x4匹配不到新的，抢夺失败，-> x5,null
```

有没有说清楚？就两步：
1. 根据关联表直接建立关系
2. 如果当前`C`匹配的对象已经被`B`匹配过了，那么尝试把它抢过来：
  * `B`去找别的匹配
    * 找到了(`A`)就建立新的匹配
      * 如果新的匹配(`A`)也已经被别人(`D`)匹配了，那么那个“别人(`D`)”也放弃当前匹配去找别的（*递归警告*）
    * 如果找不到新的匹配，那么`C`抢夺失败，递归中的`D`也同理，失败向上冒泡

注意递归怎么写代码就能写出来了：
```python
nx, ny = relationship_matrix.shape    # 6个x，7个y

# 如果x0与y0关联，x3也与y0关联，那么x0去找新的匹配时，需要把y0过滤掉
# 同理x0如果找到下一个y2，y2已被x2关联，那么x2找新的匹配时[y0, y2]都需要过滤掉
# 我们把这个数组存为y_used
y_used = np.zeros((ny,), dtype=bool)  # 存y是否连接上
path = np.full((ny,), -1, dtype=int)  # 存x连接的对象，没有为-1

def find_other_path_and_used(x):
    for y in range(ny):
        if relationship_matrix[x, y] and not y_used[y]:
            y_used[y] = True        # 处于争夺中的y，需要打标，在后续的递归时要过滤掉
            if path[y] == -1 or find_other_path_and_used(path[y]):
                path[y] = x         # 直接连接 和 抢夺成功
                return True
    return False                    # 抢夺失败 和 默认失败

for x in range(nx):
    y_used[:] = False  # empty
    find_other_path_and_used(x)

for y, x in enumerate(path):
    if x != -1:
        print(x, y)
```

真的写代码实现的时候，难点反而是`y_used`这个，第一遍代码没考虑这一点，导致递归的时候每次都从$y_0$开始而出现死循环，意识到后把处于争抢状态中的`y`打个标就好了。

scipy中有一个算法实现了Hungarian algorithm：
```python
from scipy.optimize import linear_sum_assignment

# relationship_matrix是代价矩阵
# 所以我们要代价越小越好，就用1来减
rows, cols = linear_sum_assignment(1-relationship_matrix) 
list(zip(rows, cols))
```
```
[(0, 0), (1, 1), (2, 6), (3, 2), (4, 3), (5, 4)]
```

为什么与上面不一样呢？
1. （0，0），（1，1）的匹配显然不是我们实现的后来者优先
2. 他把行看成是工人，列看成是任务，每个工人总要分配个任务，所以(5,4)这种代价矩阵里没有的关联它也做出来了，目的只是让“总代价”最小
```
(1-relationship_matrix)[rows, cols]  # 总代价为1
```
```
array([0, 0, 0, 0, 0, 1])
```
从它的名字也能看出来，它是理解为`指派问题`的(`assignment`)
