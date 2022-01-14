---
layout: post
title: Mac远程Windows-10里用Anaconda装的Jupyter-lab
slug: Mac远程Windows-10里用Anaconda装的Jupyter-lab
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

家里台式机配置比笔记本好多了，但又习惯了苹果本，怎么在小本本上直接跑windows上的jutyper呢？

首先，给Windows 10 装上[OpenSSH](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)

如果你不是用的Anaconda等虚拟环境而是把python和jupyter lab装在了本机以及写在了path里，理论上你用ssh连上windows后在shell里直接`jupyter lab`就好了，可是我是用了Anaconda的，ssh进去以及windows自身的命令行环境里都是执行不了conda和jupyter的

>可能仅仅只是path的原因，但应该没这么简单，考虑到端口转发已经能实现我的目的了，就不深究了。

这时使用`ssh`的本地端口转发功能可以达到目的：
```bash
$ ssh -L 2121:host2:21 host3
```
即把`host3`的端口`21`转发到`host2`的2121上去，当然，大多数情况下`host2`就是本机，那么`localhost`就好了：
```bash
$ ssh -L 8000:localhost:8889 windows-server
```

当然，`8889`是你在windows上运行`--no-browser`的jupyter lab设定的端口：
```bash
jupyter lab --no-browser --post=8889
```
