---
layout: post
title: add_subplots方法传递额外参数
slug: add_subplots方法传递额外参数
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - AI
---

用matplotlib绘制3D图，最快的用法
```python
ax = plt.axes(projection='3d')

此后即可  `ax.plot`, `ax.scatter`等，具体用法请翻阅文档
```
其次，这样：
```python
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
```
而我喜欢要同时绘多栏图的时候喜欢用`plt.subplots`方法，却发现传不进`projection`参数，仔细看文档，是支持用`subplot_kw`来为它添加的subplots来进行属性设置的，这样可以保持外层api干净而不必把subplots的所有属性都复制一遍
```python
fig, axs = plt.subplots(nrows=2, ncols=3, **{"subplot_kw": {'projection': '3d'}})
# 或
fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw=dict(projection='3d'))
```
看你自己习惯了。
![](../assets/1859625-3987a0af6615f30e.png)
