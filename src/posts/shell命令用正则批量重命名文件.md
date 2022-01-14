---
layout: post
title: shell命令用正则批量重命名文件
slug: shell命令用正则批量重命名文件
date: 2020-11-14 00:00
status: publish
author: walker
categories: 
  - Skill
tags:
  - shell
  - rename
---

又是用shell来操作文件的问题.

我下了老友记的全集, 结果在NAS里死活匹配不出3季以后的剧集信息, 因为打包来源相同, 一直没深究, 只当是刮削工具做得不好, 今天才发现从第4季开始, 所有的文件名格式都错了, 如:
```swift
Friends.s10.06.2003.BDRip.1080p.Ukr.Eng.AC3.Hurtom.TNU.Tenax555.mkv
```
中的`s10.06`应为`s10.e06`, 那么改对不就是了么. 又是批量任务啊, 这次的需求从上次的批量移动文件变成了批量修改文件名. 

事实上`mv`其实也是重命名工具, 奈何这次的规则稍微复杂, 我还是想要用正则来匹配, 一番搜索, 找到了`rename`这个工具. 网上的相关文章似乎有点旧, 跟今天我Homebrew下来的的文档有出入, 因此也就没看网上的文档了, 建议自己看一下官方文档, 还自带了`tutorial`和`cookbook`, 很良心啊, 看完基本自己就会了. 

新版的`rename`工具把选项分为了`switch`和`transforms`, 自然文档也就成了: 

```c-sharp
rename [switches|transforms] [files]
```

既然都推荐你们看文档了, 我了不一一介绍了, 挑几个需要注意的讲, 最后再揭晓我是如何完成这次批量重命名的.  

##debug
`-n`这个`switch`可以显示本次命令将被如何执行, 而不真正执行, 这非常像上一篇文章里介绍`xargs`时的`-p`, 在`rename`的语境里, 它叫`dry-run`. 总之我就是通过这个学习的, 非常有用.

##替换
新版`rename`加了很多像去头啊, 去尾啊, 加前缀啊, 加尾缀啊, 去空白啊, 变大小写啊等等的选项, 这个去读文档, 执行一些简单且明确的任务用这些`switch`和`transforms`比自己去构建正则要来的简单, 这也是作者把这些小功能全提取出来的目的吧. 由于我的目标是正则, 着重关注`-s`这个`transform`.  

假设有文件`abc123.mp3`和`abc456.mp3`, 以下命令均加了`-n`, 以便直接看输出

```python
#替换
$rename -n -s abc mmm *
$rename -n 's/abc/mmm/' *
#以上两句只是展示两种写法/格式
#输出:
'abc123.mp3' would be renamed to 'mmm123.mp3'
'abc456.mp3' would be renamed to 'mmm456.mp3'

#加前缀
$rename -n 's/^/album_/' *.mp3
#输出:
'abc123.mp3' would be renamed to 'album_abc123.mp3'
'abc456.mp3' would be renamed to 'album_abc456.mp3'

#演示一次错误的加前缀方式
$rename -n 's/^/album_^/' *.mp3
#输出:
'abc123.mp3' would be renamed to 'album_^abc123.mp3'
'abc456.mp3' would be renamed to 'album_^abc456.mp3'
#看到了吧? 直接把^给替换了, 而不是插入

#去后缀
$rename -n 's/\.mp3//' *.mp3
#输出:
'abc123.mp3' would be renamed to 'abc123'
'abc456.mp3' would be renamed to 'abc456'

#分组
$touch AA.S01.12.mkv AA.S01.13.mkv AA.S01.14.mkv
#这次把文件搞复杂点, 假定有如上三个文件, 我们要把12改为E12, 以此类推
$rename -n 's/\.(\d{2})\./\.E$1\./' *.mkv
#输出:
'AA.S01.12.mkv' would be renamed to 'AA.S01.E12.mkv'
'AA.S01.13.mkv' would be renamed to 'AA.S01.E13.mkv'
'AA.S01.14.mkv' would be renamed to 'AA.S01.E14.mkv'
```

看到最后一个例子是不是发现我的目标已经达到了? 我没有深入研究, 只是简单的根据实际情况把前后带点符号, 中间夹了两位数字的提取了出来, 加了字母`E`, 可能还有更简便的办法, 但我看到输出, 就急急测试去了, 果然等待数秒后, 文件全部重命名成功.

##递归
当然没那么简单, 因为4-10季的内容在各自的文件夹里, 如何递归呢? 看过我[上一篇文章](https://www.jianshu.com/p/6fab4aedc07e)的人可能会想到我又去借管道和`xargs`了吧? 这次得益于我提前读了文档, 里面也有介绍, 它还能直接应用`find`过来的结果, 还不需要像`xargs`一样给个占位, 应该是作者直接做的支持, 所以我的最终命令是这样的:

```swift
$find . -name "*.mkv" -print0 | rename -n 's/\.(\d{2})\./\.e$1\./'
```
>是的, 肯定要先`-n`看看有没有操作失误, 文件出问题就麻烦了(建议先复制一份).
此外, 因为用的是管道, 所以最后的`[files]`参数就不需要了, 我之前就是疏忽了, 复制过来时留着前面做测试的`*.mkv`尾巴, 看到出错提示才意识到.

> 2021/4/22
>我又来批量重命名的时候，发现`-print0`加上反而不行了，也就是说把带了换行符的`find`输出直接送到`rename` 里面，反而能成功，拼成一行送进去的不行，不知道上次是怎么成功的。

so far so good.

##吐槽
简书的代码块, 预览里很好看, 发布出去千奇百怪, 是什么鬼, 为了给代码着色, 我不得不在代码语言标识上乱写一通(反正写bash是不着色的)

-----

## Bonus
不小心看到关于`mv`的[这个技巧]([https://news.ycombinator.com/item?id=22860140), 如果改动的只是文件名的一小部分, 比如在`10`前面加个`e`变成`e10`, 这么做就可以了
```swift
mv Friends.s06.{,e}10.1080p.x265.mkv
```
而不需要
```swift
mv Friends.s06.10.1080p.x265.mkv Friends.s06.e10.1080p.x265.mkv
```

原文里面有两个例子, 一目了然
```css
mv foo-bar-{baz,quux}.txt
mv foo-bar{,-baz}.txt
```
以上显示的是更改和添加, 显然,你也可以猜到删除的用法, 看起来跟rename用法类似
```css
mv foo-bar{-baz,}.txt
```

当然这个贴子有很大的争论, 感兴趣可以看看.
