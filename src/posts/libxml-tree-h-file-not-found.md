---
layout: post
title: libxml-tree-h-file-not-found
slug: libxml-tree-h-file-not-found
date: 2018-09-22 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - libxml2
---

stackoverflow用户对添加libxml2库表现出了极大的抱怨，原因在要把它好好地添加进去实在是太复杂了。

我就是因为出现了'libxml/tree.h file not found’错误，才发现的这篇贴子，照着做，错误就消除了，备注如下：

原始地址：[http://stackoverflow.com/questions/1428847/libxml-tree-h-no-such-file-or-directory](http://stackoverflow.com/questions/1428847/libxml-tree-h-no-such-file-or-directory)

见e.w. parris的答案

Adding libxml2 is a big, fat, finicky pain in the ass. If you're going to do it, do it before you get too far in building your project.

You need to add it in two ways:

## 1. Target settings
Click on your target (not your project) and select Build Phases.

Click on the reveal triangle titled Link Binary With Libraries. Click on the + to add a library.

Scroll to the bottom of the list and select libxml2.dylib. That adds the libxml2 library to your project.

>注: xcode 7 以后, `.dylib`文件变成了`.tbd`文件, 想要引用`.dylib`文件, 点`add others` → `cmd+shift+G`→type `/usr/lib` → find `libxml2.dylib` or `libxml2.2.dylib`

## 2. Project settings
Now you have to tell your project where to look for it three more times.

Select the Build Settings tab.

Scroll down to the Linking section. Under your projects columns double click on the Other Linker Flags row. Click the + and add -lxml2 to the list.

Still more.

In the same tab, scroll down to the Search Paths section.

Under your projects column in the Framework Search Paths row add /usr/lib/libxml2.dylib.

In the Header Search Paths and the User Header Search Paths row add $(SDKROOT)/usr/include/libxml2.

In those last two cases make sure that path is entered in Debug and Release.

## 3. Clean
Under the Product Menu select Clean.

Then, if I were you (and lets face it, I probably am) I'd quit Xcode and walk away. When you come back and launch you should be good to go.
