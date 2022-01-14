---
layout: post
title: Windows下用AHK来映射Mac常用快捷键
slug: Windows下用AHK来映射Mac常用快捷键
date: 2019-03-17 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - macos
  - windows
  - ahk
---

最近用Windows做了一段主力机, 别的问题倒没啥, Windows下有啥干不了的呢, 哪怕鼠标滚轮方向相反, 大量快捷键不同等问题都能在非常短的适应后就不会对自己造成什么困找, 问题是我还得在Windows和Mac环境切来切去, 这样频繁地按错快捷键再纠正也挺烦人的, 决定一劳永逸地解决这个问题, 经过反复搜索和比较, 还是一个以前用Windows的时候就一直敬而远之的软件成了我的首选:AutoHotKey(AHK).

以前不想用它就是因为要写脚本, 没有大量的需求的情况下我并不想去通读一遍那并不算短的文档, 这次不同了, 需要定制一堆快捷键, 感觉这个付出是值得的, 一番文档和实验之后, 初步得到如下脚本:
(特别惊喜的是, 之前要改变鼠标滚轮的方向要改注册表, 这样你每次换了个usb口插鼠标还得再改一次, 而在ahk下都是一句话的事...是不是有点佩服这个才百来k的软件了?)

```c
;----------Application

; hyper
^`::send #4

;----------Mac Key Remap

; nature scroll
;WheelUp::send {WheelDown}
;WheelDown::send {WheelUp}

; cmd+c,v,x
#c::send ^c
#v::send ^v
#x::send ^x
; 用win+shift+x来替换原来的win+x
#+x::send #x

; clipboard history
; 用win+shift+v来替换原来的win+v(剪贴板历史记录)
#+v::send #v

; save
#s::send ^s

; undo, redo
#z::send ^z
#+z::send ^+z

; selection
#+Left::send +{Home}
#+Right::send +{End}
#+Up::send +{Up}
#+Down::send +{Down}

; skip word using alt+Arrow
!Left::send ^{Left}
!Right::send ^{Right}

; switch win+tab & alt+tab
releaseAltKey() {
  if GetKeyState("LWin") {
    sleep 100
    releaseAltKey()
  }
  else
    send {alt up}
}
#Tab::
send {alt down}{Tab}
releaseAltKey()
return

!Tab::send #{Tab}

; switch ctrl+w & cmd+w
#w::send ^{w}
; 用win+shift+w替换win+w
#+w::send #{w}

; new tab
#t::send ^t
#+t::send ^+t

; search
#f::send ^f

; cmd+a
#a::send ^a

; cmd+n
#n::send ^n
```

我这里不是教程, 就不解释每一行的原理了, 就我个人而言最最最常用的复制粘贴切换程序等等都映射了, 唯一不完美的地方就在于把热键对换我没研究透(比如ctrl+x 和 win+x对换), 有三处热键对换, 但只有一处成功(alt+tab 和 win+tab互换), 其它两处只能把其中一个热键换成别的, 而不能互相替换, 在官方论坛发贴目前还没人答复我.

多说一句, 最前面的`Application`部分我贴代码的时候没删, 也是为了介绍一种打开程序的快捷方法的思路, 一般人会用脚本检测程序是否打开, 比如`邮件`应用, 如果打开, 则将窗口激活, 如果没有, 则打开程序, 如果当然已是激活状态, 则把它最小化. 而我发现如果一个程序在底部状态栏里的话, 用win+数字键激活它也可以达到上述三个目的, 比如我的`hyper`应用排在第四个, 那么我只要反复按`win+4`就能在上述三种状态间切换, 比写代码完成检测, 打开, 激活, 最小化等要来得简单的多. 

期间还有一个收获, mac下多桌面的切换非常简单, ctrl+左右方向键即可, 或者用四根手势轻扫触摸板, 而windows下则点taskview图标, 或者用win+tab激活, 然后再用鼠标选择(总的来说是 Win + Ctrl + D / F4 / ← / → 这些组合), 怎样都没有mac的方案来得优雅(就像mac下没有一个窗口管理程序有windows自带的优雅一样^_^), 发现有人写了一个脚本检测鼠标位置, 如果到了最右侧或最左侧, 就直接把下一个桌面拉出来, 原本没打算分享只是自己用用, 因此出处没保存, 抱歉, 不过类似的脚本网上很多, 还有很多很精细的, 这一个已经算很简单粗暴了, 可以抱着学习的目的了解下, 毕竟这个脚本是以毫秒级别在时刻监视鼠标位置, 这是没必要的性能消耗吧:
```c
;----------------屏幕热区
;这个是设置鼠标坐标的相对位置，本例是相对雨整个桌面
CoordMode, Mouse ,Screen

#Persistent
;这个设置了获取鼠标信息的频率，数值越小边缘热区越灵敏
SetTimer, WatchCursor, 300
return

WatchCursor:
GetKeyState, state, LButton 
MouseGetPos, xpos, ypos, id, control 
;若要重设边缘热区的范围请，把下一行的 ; 号去掉，就会在鼠标位置显示鼠标的坐标，根据坐标修改以下数值
;ToolTip,x:%xpos% y:%ypos% state:%state%
if(state = "U" ){
    ;y方向的范围
    if(ypos > 150 and ypos < 1000){
        ;x方向的范围
        if(xpos = 2879){
            sleep 650
            Send ^#{Right}
            MouseMove, 2870, ypos
        }else if(xpos = 0){
            sleep 650
            Send ^#{Left}
            MouseMove, 20, ypos
        }
    ;显示所有虚拟桌面的热区
    }else if(xpos = 0 and ypos = 0){
        Send #{Tab}
        MouseMove, 10, 10
    }
}
return
```
注释很明确了, 自己读, 鼠标检测的热区是代码写死的, 解除显示鼠标位置的注释, 然后你把代码改成适合你的屏幕分辨率的数值即可
