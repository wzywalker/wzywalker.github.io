---
layout: post
title: 关于@synthesize
slug: 关于@synthesize
date: 2019-08-16 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
	- synthesize
	- dynamic
	- property
---

首先, `@synthesize myLocalVar = _myLocalVar;` 这句话是显式帮你省掉了一个 `getter` 方法和一个 `setter` 方法. 两个方法长什么样不赘述.

其次, 从某个版本的 Xcode 开始, 你连 `@synthesize` 这句话也不需要写了, 但是请注意, 这只是一个 IDE 的特性. 你不需要手动合成, 不代表 `@synthesize` 不作用了, 仅仅是让**你**能少写这一句话, 而 Xcode 帮你补全了.

再次, @synthesize 仅仅是一个 [clang 的 Objective-C 语言扩展](http://clang.llvm.org/docs/LanguageExtensions.html#objective-c-autosynthesis-of-properties) (`autosynthesis of properties `), 然后`clang`恰好是 Xcode 的默认编译器. 也就是说, 如果你换成了 `gcc`, 那么这个特性也就不复存在了. 

基于上述, 如果你使用了自己的文本编辑器, 然后用自己用 `clang` 从命令行编译, `@synthesize` 这一句话是需要自己写的.

最后, 有如下例外

1. 对于 `readwrite` 类型的属性, 你自行实现了 `getter` 和 `setter`
2. 对于 `readonly`  类型的属性, 你自行实现了 `getter`
	以上两种情况, 你一旦自行实现了对应的 `getter` 或 `setter`, 对于本文的`myLocalVar`例子, 你将发现 `_myLocalVar`没有了, 意味着你需要`@synthesize`一下.
3. `dynamic`与`synthesize`是互斥的
4. `@protocol`中声明的属性
5. `category`中声明的属性
6. 你覆盖(`overridden`)父类的属性时, 必须手动`synthesize`.

参考资料:
1, [When should I use @synthesize explicitly?](https://stackoverflow.com/questions/19784454/when-should-i-use-synthesize-explicitly?answertab=votes)
2, [@dynamic 与 @synthesize 关键词详解](http://suree.org/2015/09/01/Dynamic-Synthesize/)
