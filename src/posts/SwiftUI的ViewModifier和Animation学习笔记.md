---
layout: post
title: SwiftUI的ViewModifier和Animation学习笔记
slug: SwiftUI的ViewModifier和Animation学习笔记
date: 2020-10-17 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - swiftui
  - viewmodifier
  - animation
---

我们通过这篇文章练习如下几个知识点，借用的是斯坦福CS193p的课程里的demo，但是代码是我写的，也就是对着最终的效果写出我的实现的意思

## ViewModifier
![](../assets/1859625-198e1892136de13d.png)

首先，我们的卡片分为正面和背面，背面是纯色很简单，正面有一个圆环，和一张图片（实际是`emoji`，也就是`String`)，我们用`ZStack`布局好后即可：

```swift
        ZStack {
            Group {
                RoundedRectangle(cornerRadius: 10.0).fill(Color.white)
                RoundedRectangle(cornerRadius: 10.0).stroke(lineWidth: 3.0)
					CountDownCircle()  // 卡片内容1
					Text(card.content) // 卡片内容2
            }.opacity(isFaceUp ? 1.0 : 0.0)
                RoundedRectangle(cornerRadius: 10.0)
                    .opacity(isFaceUp ? 0.0 : 1.0)
        }
```

所以其实卡片内容就是emoji和圆环，因此我们就想，可不可以在我绘制好这两个内容后，调用一个通用方法就能把它变成一张卡片呢？

`ViewModifier`就是干这个的，使用语法如同：`myView.modifier(Cardify(isFaceUp:))`
提取出来的`modifier`如下：

```swift
struct Cardify: ViewModifier {
    var isFaceUp: Bool
    
    func body(content: Content) -> some View {
        ZStack {
            Group {
                RoundedRectangle(cornerRadius: 10.0).fill(Color.white)
                RoundedRectangle(cornerRadius: 10.0).stroke(lineWidth: 3.0)
                content  // 正面卡片内容
            }.opacity(isFaceUp ? 1.0 : 0.0)
            RoundedRectangle(cornerRadius: 10.0)
            .opacity(isFaceUp ? 0.0 : 1.0) // 反面卡片内容
    }
}
```

## Extension
更进一步，SwiftUI不是有很多`View.font(..).opacity(...)`的用法么，其中的`font`，`opacity`就是这些modifier，然后扩展（`extension`）给`View`的，我们也可以：

```swift
extension View {
    func cardify(isFaceUp: Bool) -> some View {
        self.modifier(Cardify(isFaceUp: isFaceUp))
    }
}
```
很简单的语法，这样最终`myView.cardify(isFaceUp:)`就能把当前内容给“**卡片化**”了


## Animation
想点击卡片翻面的时候有一个翻页效果，有一个原生的`rotation3DEffect`方法：

```swift
	myView.rotation3DEffect(
            .degrees(animatableData),
            axis: (0,1,0)  // 沿Y轴翻转，即水平翻转
            )
```

实际效果如下：
![](../assets/strip.jpg)

动画加长了，我们能看清卡片虽然有了翻面的动面，但是在开始动画的一瞬间，卡片的正面就显示出来了，我们来解决这个问题，所以我这里并不是系统讲解动画，而是在对解决问题的思路做个笔记。

> 题外话，我觉得`SwiftUI`和`Flutter`诞生时代相同，很多理念也驱同，在动画方面，也是放弃了以前要么从头画起，要么用封装得很好的有限几个动画的思路，基本上让你能用自绘+插值的方式来自己控制动画（有点类似关键帧，但关键帧的帧与帧之间也是自动的），而现在你可以完全对一个过程进行Linear interpolation，来控制动画过程（Flutter中的`lerp`函数就是干这个的，本节也有SwiftUI的类似实现）。  
比如这个翻转，`Objective-C`里直接就给你实现好了，在SwiftUI里，给的是一个最基本的几何变换，至于这上面的效果，就要你自己实现了。我认为这是对的。

按课程里的思路，卡片要么正面，要么反面，是由`isFaceUp`决定的，加入动画后，那需要这个属性在进行了50%（也就是90˚）的时候才改值

而这个属性是卡片的属性，与动画无关，所以**第一个决策**，就是把动画函数写到`ViewModifier`里面去，传进去的是卡片的属性，但是在`modifier`里，我们把它适当转化成应该转的角度（0˚或90˚）,这样在`modifier`里面不管做什么变化，都不影响外部调用者自己的语义了（方法和参数都没变）：

```swift
    init(isFaceUp: Bool) {
        // step1 这里接的是布尔值，但是我们需要把它转成对应的翻转角度
        animatableData = isFaceUp ? 0 : 180
    }
    
    // 重新定义了isFaceUp，改由翻转角度的大小决定
    // 从而解决isFaceUp在第一时间就改变的问题
    var isFaceUp: Bool {
        // step3
        animatableData < 90
    }
```

剩下的就是语法了，我们要实现一个`Animatable`的协议，与`ViewModifier`协议合并成`AnimatableModifier`，它只有一个属性，用我的话来说，就是前面提到的“动画插值”，我一直用这一个概念来理解这些新库里的动画原理，你也可以有你的理解。

总之，它需要你指定一个提供插值的来源，在这个例子中，这个来源就是`rotation3DEffect`函数，因为它会自动执行动画，显然里面的“**角度**”参数是会自己变的，我们要的就是捕捉这个“**角度**”，组合起来，看代码：

```swift
struct Cardify: AnimatableModifier {
    init(isFaceUp: Bool) {
        // step1 把参数转化成动画插值的（最终）值
        animatableData = isFaceUp ? 0 : 180
    }
    
    var isFaceUp: Bool {
        // step3 通过插值来反推正反面
        animatableData < 90
    }
    
    // step0
    // 把写死的角度变成插值
    var animatableData: Double // 这个类型是自定义的， 我们要用它来旋转角度，所以是double
    
    func body(content: Content) -> some View {
        ZStack {
            Group {
                if isFaceUp {
						// 卡片正面代码                
                } else {
                	   // 卡片反面代码
                }
        // step2
        // 课程里是有额外的角度参数，并且与animatableData进行了绑定
        // 其实为了演示插值的作用，不包装更直观
        .rotation3DEffect(
            .degrees(animatableData),
            axis: (0,1,0)
            )
    }
}
```

效果如下，其实就是解决了如何捕捉动画进度的问题，也就是`animatableData`
![](../assets/strip.jpg)

## Animation2
多一个例子，课程里每张卡片翻开就会倒计时，原本是一个大饼，我根据我的喜好改成了圆环（其实是我学教程的习惯，尽可能不去做跟教程一样的事，避免思维惰性）

那么怎么让进度条动起来呢？终于讲到了怎么手动计算**插值**，并把这组值推给动画库让它动起来的过程了。

有了上一个例子，我查看了一个`Shape`的定义，原生就conform to protocol `Animatable`的，所以我们直接添加一个`AnimatedData`试试。

```swift
var animatableData: Double   // degrees
```

>这里跟上例有一点区别，上一例动画是系统库做好的，我们只是`capture value`，所以几乎只要把那个变量摆在那，别处需要的时候直接使用就可以了，而现在我们是要主动更改这个data，从而实现绘图的不断更新，所以稍微复杂了些。

课程里把起点和终点都做成了动画参数，可能是为了演示`AnimatablePair`，而本例中起点其实是不变的，所以我实事求是，把它用最简单的方法来实现，同时，放弃对象化思维，使用动画插值的思维，不去考虑插值与原来的类的属性有什么关系，直接把插值用在需要变化的位置，这是做教学的话最直观的方案了，按我的做法，代码几乎没有变化，就多了一行和改了一行：

```swift
struct CountDownCircle: Shape {

/* 
以下注释掉的是教程的用法，保留了data与angle的关系
    var endAngle: Angle  //
    var animatableData: Double {
        get {
            endAngle.degrees
        }
        set {
            print("set: \(newValue)")
            endAngle = Angle.degrees(newValue)
        }
*/    
    // 我直观展示这个插值的用法
    var animatableData: Double   // degrees
    
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        p.addArc(center: center,
                 radius: (rect.width-10.0)/2.0,
                 startAngle: Angle.degrees(0-90),
                 endAngle: Angle.degrees(animatableData), //endAngle（教程用endAngle）,
                 clockwise: false)
        return p
    }
}
```

改造很简单，就是把告诉动画库“**结束角度**”是一个需要变动的值就好了，我们调用的时候把一个**能自己变化的值**送到这个参数里就能动起来。对调用者进行一点准备：

```swift
    @State private var animatedData: Double = 0.0
    
    private func startRemainingCountdown() {
        animatedData = 剩余进度
        withAnimation(.linear(duration: 剩余时间)) {
            animatedData = 0.0
        }
    }
```

这里做了两件事：

1. `@State`的用法，`View`是无状态的，现在我们要做动画，需要保持一些状态，这里我们保持一个“进度”的值
2. 添加了一个触发动画的函数，就是设置动画初值，设置终止值，然后通过`withAnimation`函数让它自动生成插值序列，这就是我前面提过的类似的`Flutter`的`lerp`方法，SwiftUI中没找到，但是变相提供了用系统动画来提供插值的做法。

使用就很简单了，把“**进度**”填到相应的参数位，然后选择一个时机触发，我们这里选择的是`onAppear`

```swift
		CountDownCircle(animatableData: -animatedData*360-90)
        .stroke(style: strokeStyle).opacity(0.4)
        .onAppear {
            startRemainingCountdown()
        }
```

> 需要注意的是`withAnimation`过程中对值的更改我们并不能显式捕捉，至少我试图把它显示在UI上观察它的变化是失败的，直接显示了最终值，而在接这个变化的插值的底层函数里，我能在`animatableData`的`set`方法里看到确实设置了无数的插值，暂时没有理解`withAnimation`真的有有没有直接对两个数字直接生成一系列中间值

效果如下：
![](../assets/strip.jpg)

## 后记
动画我之前写过一篇：[用CALayer绘图,添加动画和渐变](https://www.jianshu.com/p/0e4c8f0e1c23)，很明显可以看到，以前的写法仍然是黑匣子，即告诉动画库，请给我动画，动画的要求是blablabla，而现在都走了插值的路线，即把一系列值告诉你，你按照每个值直接绘图就是了，绘成啥样我自己负责。这就是我这篇文章反复强调的思路的变化，我喜欢这种思路。
