---
layout: post
title: js spread syntax
slug: js-spread-syntax
date: 2022-04-23 23:16
status: publish
author: walker
categories: 
  - javascript
tags:
  - es6
  - spread syntax
---

前段用node.js写了个解析DOM的小功能，用到了不少展开语法(`...`)，特整理记录一下

# Dictionary

```javascript
// 得到字典所有key的方法：
Object.keys(dict)
// 得到字典所有key, value的方法： 
Object.entries(dict).map(([k,v],i) => k)
// 根据字段过滤：
var filtered = Object.fromEntries(Object.entries(dict).filter(([k,v]) => v>1));
// 或者用assign和spread syntax:
var filtered = Object.assign({}, ...
Object.entries(dict).filter(([k,v]) => v>1).map(([k,v]) => ({[k]:v}))
```

# Array

```javascript
// HTMLCollection to Array
var arr = Array.prototype.slice.call( htmlCollection )
var arr = [].slice.call(htmlCollection);
var arr = Array.from(htmlCollection);
var arr = [...htmlCollection];

// remove duplicates (distinct)
let chars = ['A', 'B', 'A', 'C', 'B'];
let uniqueChars = [...new Set(chars)];
```

# String

```javascript
// 遍历一个数字的每一位
[...1e4+''].forEach((_, i) => {
        console.log(i)
});

// 首字母大写
function capitalizeFirstLetter([first, ...rest]) {
  return first.toUpperCase() + rest.join('');
}
```

很有python的风格啊