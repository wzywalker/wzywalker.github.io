---
layout: post
title: 从查找文件并移动的shell命令说开去
slug: 从查找文件并移动的shell命令说开去
date: 2020-10-13 00:00
status: publish
author: walker
categories: 
  - Skill
tags:
  - shell
  - xargs
---

一个不能更常见的需求: 从一大堆下载目录(或别的目录)里, 查找指定的文件, 并移动/复制到指定的文件夹, 如果用鼠标点开一个个的文件夹, 还有文件夹里的文件夹, 估计要累死, 当然, 即使自己不会, 也很容易查到两个shell命令:

	find path_A -name "*AAA*" -print0 | xargs -0 -I {} mv {} path_B
	find path_A -maxdepth 1 -name "*AAA*" -exec mv {} path_B \; 
都能达到目的, 第二条命令容易懂很多(`-maxdepth`去掉就是recrusive search), 去`exec`一个`mv`命令, 记得加上表示语句结束的分号就好了, 我的关注点在第一条, 趁机学学`xargs`吧.

查到[这篇文章](https://www.cnblogs.com/wangqiguo/p/6464234.html)说的不错, 先摘几个要点:

	echo 'main' | cat test.cpp
这条命令并不会把`main`输出, 因为管道确实将其作为标准输入给了`cat`命令作为标准输入, 但因为有了`test.cpp`这个命令行参数, `cat`命令就没有去读标准输入的参数了.   
其实基本上linux的命令中很多的命令的设计是先从命令行参数中获取参数，然后从标准输入中读取，反映在程序上，命令行参数是通过main函数`int main(int argc,char*argv[])`的函数参数获得的，而标准输入则是通过标准输入函数例如C语言中的scanf读取到的。他们获取的地方是不一样的。例如：

	echo 'main' | cat
这条命令中cat会从其标准输入中读取内容并处理，也就是会输出 'main' 字符串。echo命令将其标准输出的内容 'main' 通过管道定向到 cat 的标准输入中。

	cat
如果仅仅输入cat并回车，则该程序会等待输入，我们需要从键盘输入要处理的内容给cat，此时cat也是从标准输入中得到要处理的内容的，因为我们的cat命令行中也没有指定要处理的文件名。大多数命令有一个参数`-`如果直接在命令的最后指定 `-`则表示从标准输入中读取，例如：

	echo 'main' | cat -
这样也是可行的，会显示 'main' 字符串，同样输入	`cat -`直接回车与输入 `cat`直接回车的效果也一样，但是如果这样呢：

	echo 'main' | cat test.cpp -
同时指定test.cpp 和 - 参数，此时cat程序会先输出test.cpp的内容，然后输出标准输入'main'字符串，如果换一下顺序变成这样：

	echo 'main' | cat - test.cpp
则会先输出标准输入'main'字符串，然后输出test.cpp文件的内容。如果去掉这里的`-`参数，则cat只会输出test.cpp文件的内容。另外如果同时传递标准输入和文件名，grep也会同时处理这两个输入，例如：

	echo 'main' | grep 'main' test.cpp -
此处同上, 如果不加`-`, 则只会在test.cpp中搜索"main", 加了`-`, 则会在文件和标准输出中都检查关键字.

另外很多程序是不处理标准输入的，例如`kill`,`rm`这些程序如果命令行参数中没有指定要处理的内容则不会默认从标准输入中读取。所以：

	echo '516' | kill
这种命里是不能执行的。

	echo 'test' | rm -f
这种也是没有效果的。

有时候我们的脚本却需要`echo '516' | kill`这样的效果，例如`ps -ef | grep 'ddd' | kill`这样的效果，筛选出符合某条件的进程pid然后结束。这种需求对于我们来说是理所当然而且是很常见的，那么应该怎样达到这样的效果呢。有几个解决办法：

	kill `ps -ef | grep 'ddd'`    
这个时候实际上等同于拼接字符串得到的命令，其效果类似于`kill $pid`

	for procid in $(ps -aux | grep "some search" | awk '{print $2}'); do kill -9 $procid; done   
其实与第一种原理一样，只不过需要多次kill的时候是循环处理的，每次处理一个

	ps -ef | grep 'ddd' | xargs kill  
OK，使用了`xargs`命令，铺垫了这么久终于铺到了主题上。`xargs`命令可以通过管道接受字符串，并将接收到的字符串**通过空格分割成许多参数**(默认情况下是通过空格分割) 然后将参数传递给其后面的命令，作为后面命令的命令行参数

###xargs与管道的区别

```
echo '--help' | cat
echo '--help' | xargs cat
```	
第一句输出`--help`, 第二句相当于执行了`cat --help`, 所以管道是把前面的输出当成后面的输入, 而`xargs`则是把前面的输出当成了后面的命令行参数.

`xargs`的命令参数可以查我给的引用原文, 说得详细且有实例, 或者看下面的简单介绍:

```
-0，--null：以\0作为分隔符，接受到的特殊字符将当作文本符号处理；  
-d：指定分段的分隔符，默认分隔字符为空白字符；
-a，--arg-file=file：指定命令标准输入的来源文件；
-e'FLAG' 或者-E 'FLAG'：指定一个终止符号，当xargs命令匹配到第一个FLAG后，停止传递，并退出命令；
-p：每当xargs执行一个分段时，询问一次用户是否执行；
-t：表示先打印执行的命令再输出；
-n NUM：表示一个分段包含的参数个数，参数之间以分隔符隔开，默认是将所有的参数当作一个分段输出；
-i：用于将分段分批传递给其后的{}进行输出，分段会替换{}所在的位置进行输出；
-I "FLAG"：可指定分段的替换符号，分段会分批替换到符号所在的位置进行输出执行；
-L：指定每次执行的最大的非空行的行数；
```
我们来说回"查找并移动"这个原始需求.

首先, 前面铺垫的那么多`-`与标准输入的内容其实与`find`命令并无多大关系. 我们看这里面用到的三个参数

### -print0
用过`find`都知道它的结果是以换行符分隔的, 而加上`-print0`选项则可以把它换成`\0`(其实就是`NUL`)来分隔. 嗯, 不是空格, 但是至少变成了一行, 有点命令行参数的意思了吧?

### -0
就是`--null`, 以`null`为分隔符, 因为我们在前面设置`find`的输出为`null`, 这里当然要设置相应的分隔符. 如果仔细读了前面的参数表, 会发现其实它就是`-d '\0'`的简化版.

### -I
这个命令的英文说明看得我云里雾里, 一贯的不说人话风格, 我还是用一个实例来说明它的用法吧

我在一个目录里建了几个文件, 用`find`把它找出来并用`xargs`把它`echo`出来:

```
$find . -name "*.txt" -print0 | xargs -p -0 echo
echo ./c.txt ./b.txt ./a.txt?...y
./c.txt ./b.txt ./a.txt
```
	
注意, 我加了一个`-p`参数, 这是为了在执行命令前先把命令打印出来, 这样一来你有机会检查生成的命令最终是不是你想要的, 另一方面也能检查你的命令是否执行了多次.

根据上面的演示, 我们发现一个问题, 就是如果是执行`mv file path/`这样的命令, 也就是说我们需要在命令**中间**插入管道过来的参数, 是不行的, 似乎应该用占位符.

反向学习, 我们既然已经知道了`-I replstr`是正确答案, 那就尝试一下吧:

```
$find . -name "*.txt" -print0 | xargs -p -0 -I {} echo {} "HELLO"
echo ./c.txt HELLO?...y
./c.txt HELLO
echo ./b.txt HELLO?...y
./b.txt HELLO
echo ./a.txt HELLO?...y
./a.txt HELLO
```

首先, 我们发现, 我们成功地在`echo`和`HELLO`间插入了管道过来的参数, 其次, 它还把参数用分隔符自行拆开了一次执行一个(又有点类似于添加了`-n 1`的选项的意思).

现在我们明白了, 网上查到的那条命令最终就是执行了N次`mv FILE /path`, 这就是`-I {}`.

Furthermore, 我们把标准答案里那高大上的`{}`换一下如何?

```
$ find . -name "*.txt" -print0 | xargs -p -0 -I 'M' echo 'M' "HELLO"
echo ./c.txt HELLO?...y
./c.txt HELLO
echo ./b.txt HELLO?...y
./b.txt HELLO
echo ./a.txt HELLO?...y
./a.txt HELLO

$ find . -name "*.txt" -print0 | xargs -p -0 -I M echo M "HELLO"
echo ./c.txt HELLO?...y
./c.txt HELLO
echo ./b.txt HELLO?...y
./b.txt HELLO
echo ./a.txt HELLO?...y
./a.txt HELLO
```

这里我分别用了`'M'`和`M`, 都不影响其作为占位符的作用, 不要被那故弄玄虚的`{}`给迷惑了. 之所以用`{}`应该还是它更好被辨识和表义, 并不是大括号本身是什么语法.
