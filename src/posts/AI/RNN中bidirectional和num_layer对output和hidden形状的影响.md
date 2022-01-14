---
layout: post
title: RNN中bidirectional和num_layer对output和hidden形状的影响
slug: RNN中bidirectional和num_layer对output和hidden形状的影响
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - AI
---

## Batch first
首先，我们要习惯接受`batch_first=False`（就是默认值）的思维，因为NLP中批量处理句子，是每一句取第一个词，第二个词，以此类推。
按我们习惯的把数据放在同一批（即`batch_first=True`）的思路虽然可以做到（善用切片即可），但是绕了弯路。但是如果第1批都是第1个字，第2批全是第2个字，这会自然很多（**行优先**）。

所以至少`Pytorch`内部，你设了True，内部也是按False来处理的，只是给了你一个语法糖（当然你组织数据就必须按True来组织了。

看个实例：

![](../assets/1859625-9aea1f2ec2540d06.png)

1. 假定批次是64，句长截为70，在还没有向量化的数据中，那么显然一次的输入应该为(70x64)，批次在第2位
2. 注意第一行，全是2，这是设定的`<bos>`，这已经很好地表示了在行优先的系统里（比如`Matlab`就是列优先），会自然而且把**每句话**的第一个词读出来的设定了。
```
# 我用的torchtext的Field进行演示， SRC是一个Field
[SRC.vocab.itos[i] for i in range(1,4)]  
['<pad>', '<bos>', '<eos>']
```
3. 可见，2是开始，3是结束，1是空格（当然这是我设置的）
4. 同时也能注意到，最后一行有的是3，有的是1，有的都不是，就说明句子是以70为长度进行截断的，自然结束的是3，补`<pad>`的是1，截断的那么那个字是多少就是多少
5. 竖向取一条就是一整句话，打印出来就是箭头指向的那一大坨（共70个数字）
6. 对它进行`index_to_string`(itos)，则还原出了这句话
7. nn.Embedding做了两件事：
  * 根据vocabulary进行one-hot（稀疏）$\rightarrow$ 所以你要告诉它词典大小
  * 然后再embedding成指定的低维向量（稠密）
  * 所以70个数字就成了70x300，拼上维度，就是70x64x300

既然讲到这了，多讲两行，假定hidden_dim=256, 一个`nn.RNN`会输出的`outputs`和`hidden`的形状如下：
```
>>> outputs.shape
torch.Size([70, 64, 256])
>>> hidden.shape
torch.Size([1, 64, 256])
```
1. 即300维进去，256维出来，但是因为句子有70的长度，那就是70个output，hidden是从前传到后的，当然是最后一个
2. 也因此，如果你不需要叠加多层RNN，你只需要最后一个字的output就行了`outputs[-1,:,:]`, 这个结果送到全连接层里去进行分类。

## 自己写一个RNN

其实就是要自己把上述形状变化做对就行了。就是几个线性变换，所以我们用`nn.Linear`来拼接:
1. input: 2x5x3 $\Rightarrow$ 5个序列，每一个2个词，每个词用3维向量表示
2. hidden=10, 无embedding，num_class=7
3. 期待形状：
  * output: 2x5x7
  * hidden:1x5x10

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 2)
        # input shape: (2, 5, 3)
        # hidden shape: (2, 5, 10)
        # combine shape (2, 5, 13)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

hidden_size = 10
num_class = 7
input = torch.randint(0,10,(2,5,3))

rnn = RNN(input.size(2),hidden_size,num_class)
out, hid = rnn(input, torch.zeros(2, 5, hidden_size))
out.shape, hid.shape
```
output:
```
(torch.Size([2, 5, 7]), torch.Size([2, 5, 10]))
```
可见，output是一样的，hidden的形状不一样，事实上每一个字确实是会产生hidden的，但是pytorch并没有把它返出来（消费掉就没用了）。这里就pass了，我们主要是看一下双向和多层的情况下形状的变化，下面我们用pytorch自己的RNN来测试。

# num_layers

```python
import torch
import torch.nn as nn
torch.manual_seed(3)
num_layers = 3
batch_size = 2
rnn = nn.RNN(input_size=3, hidden_size=12, num_layers=num_layers, batch_first=False)
h0 = torch.randn(num_layers, batch_size, 12) # 几层就需要初始几个hidden
x0 = torch.rand(5, batch_size, 3) # input: 5x3 -> 1x12 # N个批次， 5个序列(比如5个字，每个字由3个数字的向量组成)
o, h = rnn(x0, h0) # 5个output, 一个final hidden
print('output shape', o.shape)
print('hidden shape', h.shape)
```
输出：
```
output shape torch.Size([5, 2, 12])  # 2个批次，5个词，12维度输出
hidden shape torch.Size([3, 2, 12]) # 3层会输出3个hidden，2个批次
```
加上embedding, RNN改成GRU
```python
# 这次加embedding
# 顺便把 RNN 改 GRU
vocab_size = 5
embed_size = 10
hidden_size = 8
batch_size = 3
# 要求词典长度不超过5，输出向量长度为10
emb = nn.Embedding(vocab_size, embed_size) 
# 输入为embeding维度，输出（和隐层）为8维度
rnn = nn.GRU(embed_size, hidden_size, batch_first=False, num_layers=2)
# 这次设了num_layers=2，就要求有两个hidden了
h0 = torch.rand(2, batch_size, hidden_size)
# 因为数据会用embedding包一次，所以input没有了维度要求（只有大小要求，每个数字要小于字典长度）
x0 = torch.randint(1, vocab_size, (5, batch_size)) 
e = emb(x0)
print('input.shape:', x0.shape)
print('embedding.shape:', e.shape)  # (3,4)会扩展成（3,4,10), 10维是rnn的input维度，正好
o, h = rnn(e, h0)
print(f'output.shape:{o.shape}, hidden.shape:{h.shape}')
```
```
input.shape: torch.Size([5, 3])
embedding.shape: torch.Size([5, 3, 10])
output.shape:torch.Size([5, 3, 8]), hidden.shape:torch.Size([2, 3, 8])
```
唯一要注意的变化就是input，因为embedding是把字典大小的维度转换成指定大小的维度，暗含了你里面的每一个数字都是字典的索引，所以你组装demo数据的时候，要生成小于字典大小(`vocab_size`）的数字作为输入。

## bidirectional
这次加**bidirectional**

* batch_first = False
* x (5, 3) -> 3个序列，每个序列5个数
* embedding(5, 10) -> 输入字典长5，输出向量长10 -> (5, 3, 10) -> 3个序列，每个序列5个10维向量
* hidden必须为8维，4个（num_layers=2, bidirection),3个批次 -> (4,3,8)
* rnn(10, 8) -> 输入10维，输出8维

```python
# 这次加 bidirection

vocab_size = 5
embed_size = 10
hidden_size = 8
batch_size = 3
num_layers = 2
# 要求词典长度不超过5，输出向量长度为10
emb = nn.Embedding(vocab_size, embed_size) 
# 输入为embeding维度，输出（和隐层）为8维度
rnn = nn.GRU(embed_size, hidden_size, batch_first=False, num_layers=num_layers, bidirectional=True)
# 这次设了num_layers=2，就要求有两个hidden了
# 加上双向，就有4个了，这里乘以2
# h0 = (torch.rand(2, batch_size, hidden_size), torch.rand(2, batch_size, hidden_size))
h0 = torch.rand(num_layers*2, batch_size, hidden_size)
# 因为数据会用embedding包一次，所以input没有了维度要求（只有大小要求，每个数要小于字典长度）
x0 = torch.randint(1, vocab_size, (5, batch_size)) 
e = emb(x0)
print('input.shape:', x0.shape)
print('embedding.shape:', e.shape)  # (3,4)会扩展成（3,4,10), 10维是rnn的input维度，正好
# hidden = torch.cat((h0[-2,:,:], h0[-1,:,:]),1)
o, h = rnn(e, h0)
print(f'output.shape:{o.shape}, hidden.shape:{h.shape}')
```
```
input.shape: torch.Size([5, 3])
embedding.shape: torch.Size([5, 3, 10])
output.shape:torch.Size([5, 3, 16]), hidden.shape:torch.Size([4, 3, 8])
```
可见，双向会使输出多一倍，可以用`[:hidden_size], [hidden_size:]`分别取出来，我们**验证**一下，用框架生成一个双向的GRU，然后手动生成一个正向的一个负向的，复制参数，看一下输出：

```python
from torch.autograd import Variable
import numpy as np
# 制作一个正序和反序的input
torch.manual_seed(17)
random_input = Variable(torch.FloatTensor(5, 1, 1).normal_(), requires_grad=False)
reverse_input = random_input[np.arange(4, -1, -1), :, :]

bi_grus = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)
reverse_gru = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)
gru = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)

reverse_gru.weight_ih_l0 = bi_grus.weight_ih_l0_reverse
reverse_gru.weight_hh_l0 = bi_grus.weight_hh_l0_reverse
reverse_gru.bias_ih_l0 = bi_grus.bias_ih_l0_reverse
reverse_gru.bias_hh_l0 = bi_grus.bias_hh_l0_reverse
gru.weight_ih_l0 = bi_grus.weight_ih_l0
gru.weight_hh_l0 = bi_grus.weight_hh_l0
gru.bias_ih_l0 = bi_grus.bias_ih_l0
gru.bias_hh_l0 = bi_grus.bias_hh_l0

bi_output, bi_hidden = bi_grus(random_input)
output, hidden = gru(random_input)
reverse_output, reverse_hidden = reverse_gru(reverse_input)  # 分别取[(4,3,2,1,0),:,:] -> 即倒序送入input
print('bi_output:', bi_output.shape)
print(bi_output.squeeze(1).data)
print(bi_output[:,0,1].data)                # 双向输出中的后半截
print(reversed(reverse_output[:,0,0].data)) # 反向输出
print(output.data[:,0,0])                   # 单独一个rnn的输出 
print(bi_output[:,0,0].data)                # 双向输出中的前半截
```
```
bi_output: torch.Size([5, 1, 2])
tensor([[-0.2336, -0.3068],
        [ 0.0660, -0.6004],
        [ 0.0859, -0.5620],
        [ 0.2164, -0.5750],
        [ 0.1229, -0.3608]])
tensor([-0.3068, -0.6004, -0.5620, -0.5750, -0.3608])
tensor([-0.3068, -0.6004, -0.5620, -0.5750, -0.3608])
tensor([-0.2336,  0.0660,  0.0859,  0.2164,  0.1229])
tensor([-0.2336,  0.0660,  0.0859,  0.2164,  0.1229])
```
现在你们应该知道`bidirectional`的双倍输出是怎么回事了，再来看看hidden
```python
hidden.shape, reverse_hidden.shape, bi_hidden.shape
bi_hidden[:,0,0].data, reverse_hidden[:,0,0].data, hidden[:,0,0].data
```
```
(torch.Size([1, 1, 1]), torch.Size([1, 1, 1]), torch.Size([2, 1, 1]))
(tensor([ 0.1229, -0.3068]), tensor([-0.3068]), tensor([0.1229]))
```
* 正向的输出就是单向rnn
* 反向的输出就是把数据反传的单向rnn
* 双向rnn出来的第最后一个hidden（后半截）就是反向完成后的hidden

![](../assets/1859625-5ee75456f3cd94de.png)

由打印出来的数据可知：
* 最后一个hidden，就是反向RNN的最后一个hidden（时间点在开头）
* 也是双向RNN里的第一个输出（**的最后一个元素**）
* 也是单向RNN（但是数据反传）（或者正向，但逆时序）里的最后一个输出
----
双向RNN里
* 倒数第二个hidden，是正向的最后一个hidden（时间点在结尾）
* 它也是output里面的值，它是双向输出里的最后一个的**第一个元素**

总的来说
* output由正反向输出横向拼接（所有）
* hidden由正反向hidden竖向拼接（top layer)

![](../assets/1859625-d12e030fa3d7527e.png)


