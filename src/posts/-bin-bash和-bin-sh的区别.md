---
layout: post
title: -bin-bash和-bin-sh的区别
slug: -bin-bash和-bin-sh的区别
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - Skill
---

[该文不严谨, 文末有补充]

脚本test.sh内容：
```
#!/bin/sh
source pcy.sh #pcy.sh并不存在
echo hello
```
执行./test.sh，屏幕输出为：
```
./test.sh: line 2: pcy.sh: No such file or directory
```
由此可见，在`#!/bin/sh`的情况下，source不成功，**不会运行**source后面的代码。
修改test.sh脚本的第一行，变为`#!/bin/bash`，再次执行./test.sh，屏幕输出为：
```
./test.sh: line 2: pcy.sh: No such file or directory
hello
```
由此可见，在`#!/bin/bash`的情况下，虽然source不成功，但是还是运行了source后面的echo语句。
但是紧接着我又试着运行了一下`sh ./test.sh`，这次屏幕输出为：
```
./test.sh: line 2: pcy.sh: No such file or directory
```
表示虽然脚本中指定了#!/bin/bash，但是如果使用sh 方式运行，如果source不成功，也不会运行source后面的代码。

为什么会有这样的区别呢？

junru同学作了解释

1. sh一般设成bash的软链
```
[work@zjm-testing-app46 cy]$ ll /bin/sh
lrwxrwxrwx 1 root root 4 Nov 13 2006 /bin/sh -> bash
```
2. 在一般的linux系统当中（如redhat），使用sh调用执行脚本相当于打开了bash的POSIX标准模式
3. 也就是说 /bin/sh 相当于 /bin/bash --posix

所以，sh跟bash的区别，实际上就是bash有没有开启posix模式的区别

so，可以预想的是，如果第一行写成 #!/bin/bash --posix，那么脚本执行效果跟#!/bin/sh是一样的（遵循posix的特定规范，有可能就包括这样的规范：“当某行代码出错时，不继续往下解释”）

来源: [http://www.cnblogs.com/baizhantang/archive/2012/09/11/2680453.html](http://www.cnblogs.com/baizhantang/archive/2012/09/11/2680453.html)

# 其它解释

等等,  这里就完了吗? 这里有更明确的说法

在`CentOS`里，/bin/sh是一个指向/bin/bash的符号链接: (只是在 CentOS 里哦)
```
[root@centosraw ~]# ls -l /bin/*sh
-rwxr-xr-x. 1 root root 903272 Feb 22 05:09 /bin/bash
-rwxr-xr-x. 1 root root 106216 Oct 17  2012 /bin/dash
lrwxrwxrwx. 1 root root      4 Mar 22 10:22 /bin/sh -> bash
```
**但在Mac OS上不是**，/bin/sh和/bin/bash是两个**不同**的文件，尽管它们的大小只相差100字节左右:
```
iMac:~ wuxiao$ ls -l /bin/*sh
-r-xr-xr-x  1 root  wheel  1371648  6 Nov 16:52 /bin/bash
-rwxr-xr-x  2 root  wheel   772992  6 Nov 16:52 /bin/csh
-r-xr-xr-x  1 root  wheel  2180736  6 Nov 16:52 /bin/ksh
-r-xr-xr-x  1 root  wheel  1371712  6 Nov 16:52 /bin/sh
-rwxr-xr-x  2 root  wheel   772992  6 Nov 16:52 /bin/tcsh
-rwxr-xr-x  1 root  wheel  1103984  6 Nov 16:52 /bin/zsh
```

来源: [https://github.com/qinjx/30min_guides/blob/master/shell.md](https://github.com/qinjx/30min_guides/blob/master/shell.md)
