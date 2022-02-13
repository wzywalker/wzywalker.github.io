---
layout: post
title: 《Deep Learning with Python》笔记[7]
slug: Deep-Learning-with-Python-Notes-7
date: 2021-10-12 22:17
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

# Generative deep learning

Our perceptual modalities, our language, and our artwork all have `statistical structure`. Learning this structure is what deep-learning algorithms excel at. 

Machine-learning models can learn the `statistical latent space` of images, music, and stories, and they can then` sample from this space`, **creating new artworks** with characteristics similar to those the model has seen in its training data.

## Text generation with LSTM

### Language model

很多地方都在按自己的理解定义`language model`，这本书定义很明确，能为根据前文预测下一个或多个token建立概率模型的网络。

> any network that can model the probability of the next token given the previous ones is called a language model.

1. 所以首先，它是一个network
2. 它做的事是model一个probability
3. 内容是the next token
4. 条件是previous tokens

一旦你有了这样一个language model，你就能`sample from it`，这就是前面笔记里的sample from lantent space, 然后generate了。

### greedy sampling and stochastic sampling

如果根据概率模型每次都选“最可能”的输出，在连贯性上被证明是不好的，而且也丧失了创造性，所以还是给了一定的随机性能选到“不那么可能”的输出。

因为人类思维本身也是`跳跃`的。

考虑两个输出下一个token时的极端情况：

<!-- --> | <!-- --> | <!-- --> | <!-- -->
------- | ------- | ------- | -------
纯随机，所有可选词的概率是均等的 | 毫无意义 | `max entropy` | 创造性高
greedy sampling | 毫无生趣 | `minimum entropy` | 可预测性高

实现方式：`softmax temperature`

除一个`温度`，如果温度大于1，那么温度越大，被除数缩幅度就越大（这样温差就越小，分布会更平均）-> 偏向了纯随机的概率结构（均等）

```python
import numpy as np
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)
```

写成公式
$
\frac{e^{\frac{log(d)}{T}}}{\sum e^{\frac{log(d)}{T}}}
$
这是对温度和sigmoid做了融合：
1. 一个是对目标分布取自然对数后除温度再当成e的指数给幂回去（如果不除温度，那就是先log再e，等于是原数）
2. 标准的sigmoid方程

> 这里回顾一个概念：Sampling from a space

书里大量用了这个概念，结合代码，其实就是一个predict函数，也就是说，一般人理解的“`预测，推理`”，是从业务逻辑方面来理解，作者更愿意从统计学和线性代数角度来理解。

两种训练方法：
1. 每次用N个字，来预测第N+1个字，即output只有1个(voc_size, 1)，训练的是language model
2. 每次用N个字(a, b), 来预测(a+1, b+1)， output有N个(voc_size, N)，训练的是特定的任务，比如写诗，作音乐

过程：
1. 准备数据，X为一组句子，Y为每一个句子对应的下一个字（全部向量化）
2. 搭建一个LSTM + Dense 的网络，输出根据具体情况要么为1，要么为N
3. 每一个epoch里均进行预测（如果不是为了看过程，有必要吗？我们要最后一轮的预测不就行了？）
    * 进行一次fit(就是train)，得到优化后的参数
    * 随机取一段文本，用作种子（用来生成第一个字）
    * 计算生成多少个字，就开始for循环
        * 向量化当前的种子（会越来越长）
        * predict，得到每个字的概率
        * softmax temperature，平滑概率，取出next_token
        * next_token转回文本，附加到seed后面

### DeepDream

看了一遍，不感兴趣。核心思路跟视觉化filter的思路是一样的：`gradient ascent`

1. 从对每个layer里的单个filter做梯度上升变成了对整个layer做梯度上升
2. 不再从随机噪声开始，而是从一张真实图片开始，实现这些layer里对图片影响最大的patterns的distorting

### Neural style transfer

Neural style transfer consists of applying the `style` of a reference image to a target image while conserving the `content` of the target image. 

* 两个对象：`reference`, `target` image
* 两个概念：`style`和`content`

对`B`的content应用`A`的style，我们可以理解为“笔刷”，或者用前些年的流行应用来解释：把一副画水彩化，或油画化。

把style分解为不同spatial scales上的：纹理，颜色，和visual pattern

想用深度学习来尝试解决这个问题，首先至少得定义损失函数是什么样的。

If we were able to mathematically define `content` and `style`, then an appropriate loss function to minimize would be the following:
```python
loss = distance(style(reference_image) - style(generated_image)) +
        distance(content(original_image) - content(generated_image))
```

即对新图而言，`纹理要无限靠近A，内容要无限靠近B`。

* the content loss
    * 图像内容属于高级抽象，因此只需要top layers参与就行了，实际应用中只取了最顶层
* the style loss
    * 应用`Gram matrix`
        * the inner product of the feature maps of a given layer
        * correlations between the layer's feature
        * 需要生成图和参考图的每一个对应的layer拥有相同的纹理(same `textures` at different `spatial scales`)，因此需要所有的layer参与

从这里应该也能判断出要搭建网络的话，input至少由三部分（三张图片）构成了。

**demo**

* input为参考图，目标图，和生成图（占位），concatenate成一个tensor
* 用VGG19来做特征提取
* 计算loss
    1. 用生成图和`目标图`的`top_layer`以L2 norm距离做loss
    2. 用生成图和`参考图`的`every` layer以L2 Norm做loss并累加
    3. 对生成图偏移1像素做regularization loss（具体看书）
    4. 上述三组loss累加，为一轮的loss
* 用loss计算对input(即三联图)的梯度

## Generating images

> Sampling from a latent space of images to create entirely new images

熟悉的句式又来了。

核心思想：
1. low-dimensional `latent space` of representations
    * 一般是个vector space
    * any point can be mapped to a realistic-looking image
2. the module capable of `realizing this mapping`, can take point as input, then output an image, this called:
    * generator -> GAN
    * decoder -> VAE

VAE v.s. GAN
* VAEs are great for learning latent spaces that are `well structured`
* GANs generate images that can potentially be `highly realistic`, but the latent space they come from may not have as much structure and continuity.

### VAE（variational autoencoders）

given a `latent space` of representations, or an embedding space, `certain directions` in the space **may** encode interesting axes of variation in the original data. -> inspired by `concept space`

比如包含人脸的数据集的latent space里，是否会存在`smile vectors`，定位这样的vector，就可以修改图片，让它projecting到这个latent space里去。

**Variational autoencoders**

Variational autoencoders are a kind of *generative model* that’s especially appropriate for the task of **image editing** via concept vectors. 

They’re a modern take on `autoencoders` (a type of network that aims to `encode `an input to a `low-dimensional` latent space and then decode it back) that mixes ideas from deep learning with **Bayesian inference**.

* VAE把图片视作隐藏空间的参数进行统计过程的结果。
* 参数就是表示一种正态分布的mean和variance（实际取的log_variance)
* 用这个分布可以进行采样(sample)
* 映射回original image

1. An encoder module turns the input samples *input_img* into two parameters in a latent space of representations, `z_mean` and `z_log_variance`.
2. You randomly sample a point z from the latent normal distribution that’s assumed to generate the input image, via $z = z\_mean + e^{z\_log\_variance} \times \epsilon$, where $\epsilon$ is a random tensor of small values.
3. A decoder module maps *this point* in the latent space back to the original input image.

> Because epsilon is random, the process ensures that every point that’s **close to the latent location** where you encoded input_img (z-mean) can be decoded to something **similar** to input_img, thus forcing the latent space to be continuously meaningful.

1. 所以VAE生成的图片是可解释的，比如在latent space中距离相近的两点，decode出来的图片相似度也就很高。
2. 多用于编辑图片，并且能生成动画过程（因为是连续的）

伪代码(不算，可以说是骨干代码）：
```python
z_mean, z_log_variance = encoder(input_img)
z = z_mean + exp(z_log_variance) * epsilon  # sampling
reconstructed_img = decoder(z)
model = Model(input_img, reconstructed_img)
```

VAE encoder network
```python
img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2

x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
```

1. 可见是一个标准的multi-head的网络
2. 可见所谓的latent space，其实就是transforming后的结果
3. encode的目的是回归出两个参数（本例是两个2维参数）
4. 两个参数一个理解为mean, 一个理解为log_variance

decoder过程就是对mean和var随机采样（得到z)，然后不断上采样(`Conv2DTranspose`)得到形状与源图一致的输出(得到z_decode)的过程。

1. z_decode跟z做BCE loss
2. 还要加一个regularization loss防止overfitting

> 此处请看书，演示了自定义的loss。因为keras高度封装，所以各种在封装之外的自定义的用法尤其值得关注。比如这里，自定义了loss之后，Model和fit里就不需要传Y，compile时也不需要传loss了。

> loss是在最后一层layer里计算的，并且通过一个layer方法`add_loss`，把loss和input通知给了network（如果你想知道注入点的话）

使用模型的话，就是生成两组随机数，当成mean和log_variance，观察decode之后的结果。

### GAN

`Generative adversarial network`可以创作以假乱真的图片。通过训练最好的造假和和最好的鉴别者来达到“创造”越来越逼近人类创作的作品。

* **Generator** network: Takes as input a random vector (a random point in the latent space), and decodes it into a synthetic image
* **Discriminator** network (or adversary): Takes as input an image (real or synthetic), and predicts whether the image came from the training set or was created by the generator network.

**deep convolutional GAN (DCGAN)**

* a GAN where the generator and discriminator are deep convnets. 
* In particular, it uses a `Conv2DTranspose` layer for image upsampling in the generator.

训练生成器是冲着能让鉴别器尽可能鉴别为真的方向的：the generator is trained to `fool` the discriminator。
> 这句话其实暗含了一个前提，下面会说，就是此时discriminator是确定的。即在确定的鉴别能力下，尽可能去拟合generator的输出，让它能通过当前鉴别器的测试。

书中说训练DCGAN很复杂，而且很多trick, 超参靠的是经验而不是理论支撑，摘抄并笔记a bag of tricks如下：

* We use `tanh` as the last activation in the generator, instead of sigmoid, which is more commonly found in other types of models.
* We sample points from the latent space using a `normal distribution` (Gaussian distribution), not a uniform distribution.
* Stochasticity is good to induce robustness. Because GAN training results in a dynamic equilibrium, GANs are likely to get stuck in all sorts of ways. Introducing randomness during training helps prevent this. We introduce randomness in two ways: 
    * by using `dropout` in the discriminator 
    * and by adding `random noise` to the labels for the discriminator.
* Sparse gradients can hinder GAN training. In deep learning, sparsity is often a desirable property, **but not in GANs**. Two things can induce gradient sparsity: `max pooling` operations and `ReLU` activations. 
    * Instead of max pooling, we recommend using `strided convolutions` for downsampling(用步长卷积代替pooling), 
    * and we recommend using a `LeakyReLU` layer instead of a ReLU activation. It’s similar to ReLU, but it relaxes sparsity constraints by allowing small negative activation values.
* In generated images, it’s common to see `checkerboard artifacts`(stirde和kernel size不匹配千万的) caused by unequal coverage of the pixel space in the generator. 
    * To fix this, we use a kernel size that’s divisible by the stride size whenever we use a strided `Conv2DTranpose` or Conv2D in both the generator and the discriminator.

**Train**

1. Draw random points in the latent space (random noise).
2. Generate images with generator using this random noise.
3. Mix the generated images with real ones.
4. Train discriminator using these mixed images, with corresponding targets:
    * either “real” (for the real images) or “fake” (for the generated images).
    * 所以鉴别器是`单独训练的`（前面笔记铺垫过了）
    * 下面就是train整个DCGAN了：
5. Draw new random points in the latent space.
6. Train gan using these random vectors, with targets that all say “these are real images.” This updates the weights of the generator (only, because the discriminator is frozen inside gan) to move them toward getting the discriminator to predict “these are real images” for generated images: this trains the generator to fool the discriminator.
    * 只train网络里的generator
    * discriminator不训练，因为是要用“已经训练到目前程度的”discriminator来做下面的任务
    * 任务就是只送入伪造图，并声明所有图都是真的，去让generator生成能逼近这个声明的图
    * generator就是这么训练出来的。
    * 所以实际代码是一次epoch是由train一个`discriminator`和train一个`GAN`组成.

因为鉴别器和生成器是一起训练的，因此前几轮生成的肯定是噪音，但前几轮鉴别器也是瞎鉴别的。
