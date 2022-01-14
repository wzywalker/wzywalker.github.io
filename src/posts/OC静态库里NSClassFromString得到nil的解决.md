---
layout: post
title: OC静态库里NSClassFromString得到nil的解决
slug: OC静态库里NSClassFromString得到nil的解决
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

如果你在静态库中有从类名反射回类的代码, 如下:
```
NSString *myClassStr = @"myClass";  
Class myClazz = NSClassFromString(myClassStr);  
if (myClazz) {  
    id myClassInit = [[myClazz alloc] init];  
}
```
有时候(经常)会出现得到了Class为nil的情况, 网上搜索, 一般是这么说的:
>The class object named by aClassName, or nil if no class by that name is currently loaded. If aClassName is nil, returns nil.

来自于64位系统的一个bug:
>IMPORTANT: For 64-bit and iPhone OS applications, there is a linker bug that prevents -ObjC from loading objects files from static libraries that contain only categories and no classes. The workaround is to use the -all_load or -force_load flags. -all_load forces the linker to load all object files from every archive it sees, even those without Objective-C code. -force_load is available in Xcode 3.2 and later. It allows finer grain control of archive loading. Each -force_load option must be followed by a path to an archive, and every object file in that archive will be loaded.

就我的实测
+ 首先, 你需要在你的主项目(的target)对`build setting`进行更改, 而**不是**静态库的项目!
+ 其次, `-all_load`有效, `-force_load`甚至编译都过不了
+ 最后, 结合上面, 就是在主项目(引用静态库的项目)的build setting里面搜索`other linker flags`, 然后把`-all_load`加进去就行了
