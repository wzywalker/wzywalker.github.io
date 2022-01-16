---
layout: post
title: Semi supervised Learning
slug: Semi-supervised-Learning
date: 2021-06-07 00:00
status: publish
author: walker
categories: 
  - AI
tags:
  - semi supervised learning
  - 半监督学习
---

李宏毅机器学习2021spring的家庭作业里面有一个`Semi-supervised Learning`的任务。

具体来说，就是一个图片分类的任务（11个食品类别），但只给了你几百个有标注的图片，同时，还给了你几千张没有标的图片（用来训练，而不是测试）。

思路也很简单，既然样本量过小，我们就得自己扩充样本量，但这次不是用数据增广(`Augumentation`)，而是自己造样本：

1. 用小样本训练一个模型，用这个模型来predict没有标注的图片（文本有补述）
2. 对预测输出的11个类别softmax后，观察最大值，如果大于你设定的某个threshold，比如0.68，就把该图片和最大值所映射的类别当成一组真值添加到训练集里去
3. 我用的是`torch.utils.data`里的`TensorDataset`来构建手动创建的增强数据集，然后用了`ConcatDataset`与原训练集拼接：

```python
from torch.utils.data import TensorDataset

def get_pseudo_labels(dataset, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    # Iterate over the dataset by batches.
    images = torch.Tensor([])
    targets = torch.Tensor([])
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # 在这里根据阈值判断是否保留
        # Filter the data and construct a new dataset.
        for idx, prob in enumerate(probs):
            c = torch.argmax(prob)
            if prob[c] > threshold:
                torch.cat((images, img[idx]))   # 用索引选出对应的图片
                torch.cat((targets, torch.tensor(c))) # 用最大值索引当class
            
    dataset = TensorDataset(images, targets)  # 拼成tensor dataset

    # # Turn off the eval mode.
    model.train()
    return dataset
```

使用：

```python
pseudo_set = get_pseudo_labels(unlabeled_set, model)

# Construct a new dataset and a data loader for training.
# This is used in semi-supervised learning only.
concat_dataset = ConcatDataset([train_set, pseudo_set]) # 拼接两个dataset(只要有感兴趣的两组数组即可)
train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
```

看来，所谓的半监督仍然是有监督，对于没有标注的数据，仍然要想办法用已有数据去为它打标，接下来就是普通的监督学习了。

----

最后，在实际的demo代码中，能看到并不是我最初理解的“先用小样本训练好一个模型”，再用它来过滤un-labeled样本，增广到训练集去，即对训练集的增广是一劳永逸的（像别的增广方案一样）

而是每一个epoch里面都**重新**去增广一次，这个思路更类似于GAN（生成对抗网络），`generator`和`discriminator`是一起训练的。

也所以，第一次去增广的时候，其实就是一个初始化的model，也就是说，一个比较垃圾的数据集（当然，初始化的model未必能预测出置信度高的结果，以至于并不会有太多pseudo labels进入训练集）

因此，相比较纯监督学习，假如训练集是2000条，那么整个epoch轮次里，都是2000条数据在训练；而半监督学习里，可能是200, 220, 350, 580, 1000, 1500...这样累增的样本量（随着模型越来越好，置信度应该是越来越高的），如果epoch数量不够，可能并没有在相同2000左右的样本量下得到足够的训练
