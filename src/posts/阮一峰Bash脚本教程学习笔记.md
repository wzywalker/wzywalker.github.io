---
layout: post
title: -bin-bash和-bin-sh的区别
slug: -bin-bash和-bin-sh的区别
date: 2020-07-02 00:00
status: publish
author: walker
categories: 
  - Skill
tags:
  - bash
  - sh
  - shell
---

[教程地址](https://wangdoc.com/bash/intro.html)

## 快捷键:
```
ctrl+L: 清除屏幕并移到顶部
ctrl+U: 删到行首(剪) ctrl+A: 移到行首
ctrl+K: 删到行尾(剪) ctrl+E: 移到行尾
ctrl+b, f, (左移, 右移)
alt+b,f (前, 后移一个单词)
ctrl+1 等同于clear
ctrl+d delete, ctrl+w 删到单词首
alt+t 词换位, ctrl+t 字符换位 (与前一个)
alt+d, alt+backspace 剪切至词首, 词尾
ctrl+y 粘贴
```
```
$ ’Something wrong happend’ >&2
```
解释:
>`>`代表重定向输出
`&`表示接下来的是一个文件描述符数字(_file descriptor number_)
`2`表示`stderr`
所以就是把上述字符串输出到stderr里去的意思,
如果是`$ ”str” <2` (没有&), 则会输出到一个叫2的文件里去

## echo
`-n`: 可以取消结尾的换行 
`-e`: 解释字符串里的转义符

## 扩展(通配符)
```
$ ~/walker # 表示用户目录下的walker目录
$ ~walker # 表示名为walker的目录
$ ~+ # 扩展为当前目录, 等同于 pwd
$ echo d{a, e, I, o, u}g 输出: dag, deg, dig, dog, dug # 大括号内不要有空格(否则会当成参数)
$ echo {j{p,pe}g, png} —> 嵌套
```
波浪线, 方括号的括号都是基于”**路径**”的, 如果当前路径没有匹配到对应的文件名, 则会**变成字符串**原样输出, 而大括号则不然, 是基于”**逻辑**”的, 只管扩展, 不会去探测扩展后对应的路径存不存在, 因此可能报错文件不存在. 
如`echo [a,b].txt`, 如果不存在a.txt, b.txt, 则会变成`”[a,b].txt”`这样一个输出, 而`{a,b}.txt`则**一定**会扩展成a.txt, b.txt
**例外**: 
在用`..`来扩展时, 如果系统无法理解, 则不会扩展, 如`{1..5}`会扩展成1,2,3,4,5, 但`{ab..123}`, 则会变成字符串
但是前导0不参与路径匹配: `{01…5}` # 01,02,03,04,05 (几个零都可以)
步长:{0..8..2}  (`未测试成功`) # 要打开哪个shopt开关?
活用:
```
$ echo .{mp{3..4},m4{a,b,p,v}} # 匹配了: .mp3 .mp4 .m4a .m4b .m4p .m4v
$ mkdir {2007..2009}-{01..12}  # 建了2007-2009每年12个目录
for I in {1..4}
```
注意惊叹号的使用(类似正则里的`^`)
```
$ echo ${!S*} # 返回所有以S开头的”变量名”, 如SHELL, SSH..等
```
另两种转义(`string interpolation`):
```
$ echo date is $(date) # 即包在$(…)中
$ echo date is `date` # 包在反引号中
```
但是要计算2+2, 只有`echo $((2+2))` 这种形式, 反引号就不行了

[[:alnum:]], [[:digit:]]等**预置**的字符类扩展见: [https://wangdoc.com/bash/expansion.html](https://wangdoc.com/bash/expansion.html) 很丰富, 建议详读. 

`(?, \*, +, @, !)`则为匹配的个数, 分别是(0或1, 0或多, 一或多, 一个, 非一个), 如`song@(.)mp3`等同于song.mp3,
是的, 不同于正则, 它是先规定**个数**, 再设定匹配字串
> 注: 需要打开`shopt -s extglob`

双引号碰到\$, 反引号和反斜杠都会自动扩展, 所以`echo “$SHELL”` 等同于`echo echo \$SHELL`
双引号能保留”输出”的格式, 比如 echo \`cal\`, 格式就没了, 自己试试看? 而`echo “$(cal)”`则可以保留格式:
```
$ echo "$(cal)"
      六月 2020
日 一 二 三 四 五 六
    1  2  3  4  5  6
 7  8  9 10 11 12 13
14 15 16 17 18 19 20
21 22 23 24 25 26 27
28 29 30
```
大段文字输入可以用
```
$command << token
your long inputs
token
```
等同于: `echo "your long inputs" | command` 即把echo的输出作为command的输入, 这个一般用于多行文本
>texts里面可以进行使用变量, 但是如果把token用双引号包起来就不能解释变量了.

如果只是简单字符串, 用下面更明确:
`$ command <<< ‘text’` 
如: `$ cat <<< “hello world”`, 一样等同于 `$ echo “hello world” | cat `
`<<<` 还有一个作用就是把变量值用这种方式能变成**标准输入**, 这样被”**计算**”出来的值也能用于只接受标准输入的命令了, 比如`read`

## 变量
`printenv PATH` 与 `echo $PATH`等同
解释变量中的变量, 比如`$PATH`, 不是想象中的\$嵌套: `${${myvar}}`, 应该这么用
```
$ myvar=PATH
$ echo ${!myvar}, # 即多加一个惊叹号
```
`$?`: 上一个命令的退出码(0成功, 1失败) 
`$$`: 当前Shell进程的ID
`$_`: 上一个命令的最后一个参数
`$!`: 最后一个后台执行的异步命令的进程ID
`$0`: 当前Shell的名称
`$-`: 当前shell的启动参数, `$@`, `$#` 表示脚本的参数数量
> `$?`命令除了取出上一个命令的返回值, 也可以取出上一个函数的返回值

`${varname-:value}` 取值, 如果不存在则返value, 但**不赋值**
`${varname=:value}` 取值, 如果不存在则返value, 顺便**赋值**
`${varname+:value}` 如果有值则返value(而**不是值本身**), 没有值则为空, 所以这个时候的value一般用一个标识符号就好了
`${varname?:value}` 取值, 如果不存在就**报错**并把value作为错误错误打印出来
比如 `$ filename=${$1:?”filename missing”}` 从脚本中取第一个参数作为文件名, 发现没有文件名就报错退出

变量都是字符串, 可以用`declare`来进行一些限定
```
$ declare -I v1=13 v2=14 v3=v1+v2 # 声明为integer
$ echo $v3
# 这样更快: 
$ let v=13+14  # 如果习惯了=两边加空格, 则包到引号里: $ let “v = 13 + 14”
```
## 字符串
`${#”string”}` 长度
`${varname:offset:length}` 切片(变量名_不需要_美元符号)
删除: (# 和 ## 的区别就是贪婪与否的区别)
```
$ phone="555-456-1414"
$ echo ${phone#*-}
> 456-1414
$ echo ${phone##*-}
> 1414
```
替换: `${variable/#pattern/string}` 注意, `#` 左边多了一个 `/` , 右边多了替换字串
以上, 都是从头匹配, 从**尾部**匹配把 `#` 换成 `%` 
**任意位置**匹配则换成/, 所以就成了你们最熟悉的语法:`varname/search/replace`
> 这个时候再回头看`/#` , `/%`, 不过是`/`语法的修饰符罢了(限定起始方向)

`${varname^^}`, `${varname,,}` # 转大写, 转小写

## 数值运算
逗号是求值, 如
`$ echo $((foo = 1+2, 3*4))` 输出为12, 但foo的值是3, 依次计算, 输出是逗号后面的
> `expr`命令等同于双括号: `expr 3+5` 与`$((3+5))`同义

## 行操作
Bash内置Readline库, 默认采用Emacs快捷键, 切换:
```
$ set -o vi 或 $ set -o emacs
```

## 切换目录/堆栈
不管你cd到哪个了哪个目录, 想回到cd前的目录, 用`cd -`就行了
`pushd` , `popd`则可以把目录推到堆栈里, 演示:
```
$ pushd 2
/test/1/2 /test/1
$ pushd 3
/test/1/2/3 /test/1/2 /test/1
$ pushd 4
/test/1/2/3/4 /test/1/2/3 /test/1/2 /test/1
$ dirs  # 其实每一次pushd都会把当前堆栈dirs出来
/test/1/2/3/4 /test/1/2/3 /test/1/2 /test/1
$ cd /tmp
$ dirs # 观察, cd其实只是把顶层给改了, 不会增减层级
/tmp /test/1/2/3 /test/1/2 /test/1
$ cd /usr
$ dirs  # 验算
/usr /test/1/2/3 /test/1/2 /test/1
```
现在你知道 了, cd永远只是更改顶栈, 大多数情况下, 你可以用`pushd`来替换`cd`, 这样你就有了*后退权*了
此时你再`popd`, 目录会顺利切到`/test/1/2/3`, 不管你进行过多少次`cd`, 第二层都不会变并且能直接`pop`出来
$小练习$
如果你查看堆栈, 要从第4个开始后退(0为起始), 那么可以把从3开始(不是4)的记录提到顶层来(然后再`popd`):
`$ pushd +3` (加号不可省)
> 注意, 此时0, 1, 2都还在, 只是挪到了尾巴

而`popd +3`则不是”**移动**”, 而是**删除**了, 意思是正向删除**第**3个, 如果不带`+`, 则理解为删除3**以后**的所有堆栈(即从4开始)
> 注意: 为什么要从第4个开始退要把从3开始的移到顶层呢? 
因为如下`dirs: /1, /2, /3` , 你做`popd`, 是会回到/2的
可见顶栈永远表示的是"当前"目录. 所以你自然无法**跳到**当前目录.
而`pushd`, `popd`+`数字`改动的只是堆栈表, 不是目录, 即虽然你的目录没变, 但是系统**认为**你在第三层, 这个时候再后退(`popd`)自然到了目录表里的下一层`/4`.

## 脚本
shebang行`#!/usr/bin/env bash` 的写法是为了避免`#!/bin/bash` 这种写法时bash不在bin目录
`source`命令可以 用一个点来表示: `. ~/.bash_profile`
读用户的输入: 
```
$ read firstname lastname
$ “you input: $firstname, $lastname” # 如果read后没有给变量名, 则由默认的$REPLY来取出
```
读文件:
```
while read myline  # 每次读一行
do
  echo "$myline"
done < $filename  # 注意这里特殊的传参方式, 同时, 如果不传入文件路径, 就是一个无限循环了(read)
```
存数组:
```
$ read -a varname. #  -a 参数把用户的多个输入全存到`varname`这个数组里了
```
其它有用的:
```
# -e 参数使得用户在输入的时候能用tab补全(包含所有readline库快捷键),
# 如果没有这个参数输入文本的时候是不能使用快捷键的
$ read -e -p “please input the path to the file” 
# -s 可以隐藏用户的输入, 通常用于密码
$ read -s  -p “input password”
# -p 显然就是能直接显示输入前的提示了
```

## 条件判断
`if`里面的`test`命令
`test expression`, `[ expression]`, `[[ expression ]]` 是等价的(*第三种*支持正则)  **空格不能省**
`[ -? file ]` 查看文件状态有非常多的表达式(参数), 具体参阅https://wangdoc.com/bash/condition.html 推荐阅读
for循环
`for [ test ] in list; do … done` 其中的`in list`如果省略, 则代表所有脚本参数`”$@“`:
```
$ for filename; do echo “$filename”; done
# 等同于
$ for filename in “$@“; do echo “$filename”; done
```
同理, 如果是用在**函数中**, 则等于所有函数参数.
用双括号, 变量也无需加`$`了
`for (( i=0; i<5; i+=1 )); do echo $i; done`
`case in`, 如果希望一个匹配后继续做下一个匹配(`passthrough`), 每一个case 的结尾用`;;&`而不是`;;` (多了一个&)

## select
select生成的菜单, 选择并执行命令后, 要自行在`do-done`体内用`break`退出, 否则会一直要你选择

## 数组
以下方式声明数组
```
$ names=(hatter [5]=duchess Alice), 指定了0, 5, 6, 其它为空字符串
$ mp3s=( *.mp3 )
$ declare -a ARRAYNAME
$ read -a ARRAYNAME
```
读取的时候: `$ echo ${array[1]}` 大括号不可省
`@`仍然是返回所有元素: `$ echo ${array[@]}` 但是在`for…in`中, 要把整个表达式放双引号中:
```
$ activities=( swimming "water skiing" canoeing "white-water rafting" surfing )
$ for act in “${activities[@]}”; do….; done
```
不然其中有”water”, “skiing”,”white-water", rafting”等都会被拆开(_bug吧? 字符串也拆_)
把`@`换成`*`, 加上双引号, 则会一个个字符返回
拷贝数组最方便的方法:
```
$ hobbies=( “${activities[@]}” diving ) # 顺便演示了为数组添加成员
```
直接赋值给一个数组(即没有指定索引), 则是赋给**第0个组员**, 同理, **使用**数组名也是使用的0号组员
```
# 以下@可以换成*
$ echo ${#array[@]} # #仍然用以计数, 但是如果传的是具体索引, 则返回的是对应项的字符串长度
$ echo ${!array[@]} 用以返回有值的索引 (为空的不返回) # 活用的话遍历数组更高效
$ echo ${array[@]:2:3} # 切片
$ arr+=(3 4 5) # 追加
$ unset arr[2] # 删除 , 或: 
$ arr[2]= # 或
$ arr[2]=‘’
# 以上三者等效, 
```
根据上面知识`$ arr=` 表示删除第一个成员, 但是`unset arr` 则是清空整个数组了
也可以用**字符串**做索引, 就成了字典了: 
```
$ declare -A colors  # 变成大写即可
$ colors[“red”]=“#ff0000”
```

## set命令
单独一个set会显示所有环境变量和Shell函数
以下都可以以`set -xxx` 的方式写在**脚本头**或**任何位置**, 就当一个即时开关使用吧
也可以在调用bash脚本前传入比如: `bash -eux script.sh`
`-u`: 遇到不存在的变量就报错, 而不是忽略 与 `-o nounset` 等价
`-x`: 每一个命令执行前会先打印出来 等同于 `-o xtrace`, 关闭用`set +x` (组合起来用就是一个小环境)
`-e`: 有错误就中止 等同于 `-o errexit`
`-o pipefail`: 即使在管道中, 有错也中止(-e 在管道中会失效)
`-n`: `-o noexec` 不执行命令只检查语法
`-f`: `-o noglob` 不对通配符进行文件名(路径)扩展  可用+f 关闭
`-v`: `-o verbose` 打印shell接收到的每一行输入     可用+v 关闭
$ set -euxo pipefail 一般这么四个连用

## shopt
即: shell option
同set, 直接shopt也可以列出所有参数, `-s`, `-u`分别是是打开, 关闭某个参数
`shopt 参数名`, 可直接查询该参数是否打开关闭, 但是如果是用于编程, 因为返回是字符串不好判断, 所以提供了`-q`参数(返回0/1, 分别表示打开/关闭)
```
$ if shopt -q globstar; then …; fi
```

## 除错
```
# 先看目录存不存在, 然后再进入, 然后再打印出来将要删除的文件, 
# 这是最安全的删除方法
# 否则一旦目录不存在, 不同的写法会有不同的问题
[[ -d $dir_name ]] && cd $dir_name && echo rm *  
```
如果在执行bash脚本前加入`-x`参数, 则每一条命令**执行前**都会打印出来 # 等同于`set -x`
> 或者写在脚本的shebang行里也行

每一条命令会同上一个**标识符**作前缀, 默认是`+`, 可以用`export PS4=‘$LINENO +’`这种方式自定义(比如现在就加上了行号)
$几个环境变量$
`$LINENO`: 这个变量在哪, 打印的就是这一行的行号
`$FUNCNAME`: 返回一个数组, 函数调用的名称堆栈, 最里层(即本函数)的是0
`$BASH_SOURCE`: 返回一个数组, 函数调用的脚本堆栈, 即每层调用的脚本是哪一个, 最里层(即本文件)的是0
`$BASH_LINENO`: 返回一个数组, 函数每一次被调用时在该脚本的行号, 同样也是从最里层开始
例:
```
${BASH_SOURCE[1] = main.sh  # [0] 是文件本身, 所以要[1]
${BASH_LINENO[0] = 17  # 调用来源的行号  —> 所以调用来源的行号的索引永远比调用来源(文件)的索引要小1
${FUNCNAME[0]} = hello  # 本方法(或者说”被调用的方法”)
```
上例代表在 main.sh的17行调用了hello()方法 
$小练习$ 
```
#!/bin/bash
source lv2.sh   # 引入外部脚本
function lv1method()
{
    echo ---------lv1------------
    i=0
    for v in "${BASH_LINENO[@]}"; do
        echo "bash_line_no[$((i++))]: $v"
    done
    i=0
    for v in "${FUNCNAME[@]}"; do
        echo "func_name[$((i++))]: $v"
    done
    i=0
    for v in "${BASH_SOURCE[@]}"; do
        echo "bash_source[$((i++))]: $v"
    done
    lv2method # 调用外部脚本的方法
}
```
以上脚本, 多做几次嵌套,  打印出来看看索引之间的关系
输出:
```
---------lv1------------
bash_line_no[0]: 5
bash_line_no[1]: 0
func_name[0]: lv1method
func_name[1]: main
bash_source[0]: lv1.sh
bash_source[1]: entry.sh
---------lv2------------
bash_line_no[0]: 21
bash_line_no[1]: 5
bash_line_no[2]: 0
func_name[0]: lv2method
func_name[1]: lv1method
func_name[2]: main
bash_source[0]: lv2.sh
bash_source[1]: lv1.sh
bash_source[2]: entry.sh
---------lv3------------
bash_line_no[0]: 19
bash_line_no[1]: 21
bash_line_no[2]: 5
bash_line_no[3]: 0
func_name[0]: lv3method
func_name[1]: lv2method
func_name[2]: lv1method
func_name[3]: main
bash_source[0]: lv3.sh
bash_source[1]: lv2.sh
bash_source[2]: lv1.sh
bash_source[3]: entry.sh
```

## 临时文件
安全的用法:
```
trap 'rm -f "$TMPFILE"’ EXIT  # 退出时删除临时文件)
TMPFILE=$(mktemp) || exit 1  # 用mktemp命令建立临时文件可以只有本人能读, 如果失败就退出
echo "Our temp file is $TMPFILE”
```
参数: 
`-d`: 创建的是目录
`-p`: 指定目录
`-t`: 指定模板
如 `mktemp -t aaa.XXXXXXX` 能生成`/tmp/aaa.yZ1HgZV`(与X个数相同)
`trap`是用来响应系统信号的, 如`ctrl+c`产生中断信号`SIGINT`
`$ trap -l` 列出所有信号(自己打印出来看看)
`trap`的格式: `$ trap [动作] [信号1] [信号2] ...`
`trap` 命令接的信号有如下
* HUP：编号1，脚本与所在的终端脱离联系。
* INT：编号2，用户按下 Ctrl + C，意图让脚本中止运行。
* QUIT：编号3，用户按下 Ctrl + 斜杠，意图退出脚本。
* KILL：编号9，该信号用于杀死进程。
* TERM：编号15，这是kill命令发出的默认信号。
* EXIT：编号0，这不是系统信号，而是 Bash 脚本特有的信号，不管什么情况，只要退出脚本就会产生。

如果trap要执行多条命令, 可以封装到函数里, 命令的位置写函数:`$ trap func_name EXIT`

## 启动环境
登录session依次启动如下脚本:
* /etc/profile
* /etc/profile.d # 目录下的所有.sh文件
* ~/.bash_profile # 如果有, 则中止
* ~/.bash_login # 如果有, 则中止  此为C shell 初始化脚本
* ~/.profile # Bourne shell 和 Korn shell 初始化脚本

通过`$ bash - -login` 参数, 可以强制执行以上脚本
非登录session
* /etc/bash.bashrc # 所有用户都执行
* ~/.bashrc # 当前用户的

启动参数:
`-n`: 不执行脚本, 只检查语法
`-v`: 执行语句前先输出
`-x`: 执行语句后输出该语句

`~/.bash_logout` 退出时要执行的命令
`$ include /etc/inputrc` 在`~/.inputrc`里加这一行, 可以在里面自定义快捷键

## 命令提示符
上面提到过`$PS4`能修改`set -x`时打印的每句语句前面的`+`号
命令提示符默认的`$`符号(**根用户**是`#`号)则可以用`$PS1`来修改, 怎么改参考[https://wangdoc.com/bash/prompt.html](https://wangdoc.com/bash/prompt.html)
`$PS2`表示的是输入时折行的提示符, 默认为`>`
`$PS3`表示使用select命令时系统输入菜单的提示符
