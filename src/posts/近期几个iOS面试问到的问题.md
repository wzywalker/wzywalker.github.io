---
layout: draft
title: 近期几个iOS面试问到的问题
slug: 近期几个iOS面试问到的问题
date: 2021-11-05 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - interview
---

```swift
extension Sequence where Iterator.Element: Hashable {
    func unique() -> [Iterator.Element] {
        var seen: Set<Iterator.Element> = []
        return filter { seen.insert($0).inserted }
    }
}
```

**问时间复杂度**
我也记不清是问什么了，就理解为时间复杂度吧，我一直对这些绕着走，所以当时就直言说不出来，真去了解一下也不复杂。
fitler本身就是一个遍历，n
set因为无序，它的insert直接就是常数1吧
所以是O(n)吗？

**组件化思路**
我也是脑袋短路，一时间没想明白就说不清楚了。
按我的理解，大体上还是往route/mediator里注册方法来实现解耦吧，从互相依赖，变成多对一的依赖，然后通过接口/协议等继续抽象，从对对象的依赖变成对承诺的依赖，结果就是面向约定和面向运行时编程。

**属性修饰符的含义**
atomic: 原子操作，读取相对安全， 相对noatomic（可以保证一次读取操作的完整性，但是只管getter/setter，即读写安全，但不管生命安全，比如被别的线程释放）
assign: 用于数字，布尔等简单类型，所以不担心对象在别处被释放
copy: own这个对象，ARC计数不变，自身计数为1
strong:理解为retrain，这个属性就持有所指向的对象了，ARC计数加1
weak: 不持有对象，ARC计数不变

**weak指向的对象释放时怎么处理野指针**
虽然arc没有加1，但是对象还是能找到哪些持有者用了weak, `dealloc`方法就会对它们持有的这个属性设nil

**GCD队列间的关系**
我不知道想问的是啥，难道就是说并发和串行的关系？

**设计模式的基本原则**
* 单一职责原则，这是个含糊的原则，一个方法/接口尽量只做一件事，什么一个类只能有一个引起其变化的原因，这个是应用最少的吧
* 开放封闭原则，通过抽象保持软件架构稳定（封闭），通过扩展来实现功能的拓展（开放）
* 里氏替换原则，据说是开闭原则的补充，阐述了子类继承父类时，除添加新的方法完成新增功能外，尽量不要重写父类的方法。
* 依赖倒置，这个倒是常用的，针对接口/约束/协议去编程，即依赖于抽象，这些拥有稳定的规范和契约，具体的实现是注入进来的（“依赖注入”）
* 接口隔离原则，感常见跟单一职责差不多，一个接口尽量做一件事，据说这就是跟单一职责最大的区别，它只是针对接口来说的。
* 迪米特原则，感觉组件机制就是应用的这个原则，即不同类之间不直接通信，而通过一个中间类，解耦类之间的互相依赖

**runtime的理解**
Runtime中，对象可以用C语言中的结构体表示，而方法可以用C函数来实现，其它具体的见海量博文吧

项目里可能会用到的
* 关联对象 Associated Objects
* 消息发送 Messaging
* 消息转发 Message Forwarding
* 方法调配 Method Swizzling
* “类对象” NSProxy Foundation
* KVC、KVO About Key-Value Coding

**定时器的几种实现方式，区别**
timer和gcd，后来知道了cadisplaylink，会随着屏幕的每次刷新调一次target的selector

**block为什么要copy**
block作为一个普通变量存在栈上，会随作用域消失而消失，而block的调用时机却是不定的，copy的话能自己持有一份。
这个问题一搜还是个经典面试题，想知道更多细节的自己搜吧。

**触摸事件的机制**
总的来说，由传递链把事件打包到队列里，通过hittest相关方法从windows到view到顶层subview到底层subview来逐层找第一响应者，找到就退出
而响应者就往上找能处理该事件的对象（包括自己），找到就退出。目的都是找到第一响应者。
