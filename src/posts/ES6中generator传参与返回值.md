---
layout: post
title: ES6中generator传参与返回值
slug: ES6中generator传参与返回值
date: 2020-11-13 00:00
status: publish
author: walker
categories: 
  - Front
tags:
  - es6
  - generator
---

先看两个例子, 

1,
```
function* f() {
  for(var i=0; true; i++) {
    var reset = yield i;
    if(reset) { i = -1; }
  }
}

var g = f();

document.write(g.next().value) // { value: 0, done: false }
document.write(g.next().value) // { value: 1, done: false }
document.write(g.next(true).value) // { value: 0, done: false }
```
2, 
```
function* gen(x){
  try {
    var y = yield x + 2;
  } catch (e){ 
    document.write(e);
  }
  return y;
}

var g = gen(1);
g.next();
g.throw（'出错了'）;
```
有什么区别?
第一个里传入了一个`true`参数, 第二个里传入了一个`1`参数, 目的都是期望传递给generator.
但例一演示的参数, 传过去是传给了`yield`语句本身的返回值, 即`reset`, 也就是说, 如果你没有传参, 每一次`next`方法, `reset`获取的结果都是`undefined`
例二中, 方法本身就有入参, 所以千万不要搞错了, 这种入参等于是一个`种子`, 所以只需要在实例化这个生成器的时候才需要传. 

区别就在是在生成器里传, 还是在生成器的next方法里传. 前者是给生成器赋种子值, 后者是给每个yield赋返回值
