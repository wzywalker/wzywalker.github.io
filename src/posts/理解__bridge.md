---
layout: post
title: 理解__bridge
slug: 理解__bridge
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

比较受用, 全文转载, [原文点此](https://abson.github.io/2016/08/19/Object-C-理解桥接/)

##为什么使用要使用 Object-C++

在 iOS 开发当中，难免会使用到 OC 跟 C++混编的情况，一是为了程序对负责计算性能的提高，二是因为某些三方开源库是用 C++ 来写的，这两个原因也是让我下决心学好 C++ 的因素，毕竟开源才是王道，一直只写着 OC 却不能窥其究竟，确实难受，让只能让人停留在门外，坐井观天。

##什么是桥接 ？

桥接，是 object-c 在 ARC 环境下开发出来的一种用作转换 C 指针跟 OC 类指针的一种转换技术。
当然，这种技术在 MRC 中是不存在的，也就是桥接是 ARC 的连带产物，因为 ARC 就是解放了我们程序员的双手，当然对内存的概念又淡化了，所以在 ARC 未被业界接受之前多少也是因为这个桥接让人们感觉恶心。

##桥接用到的3个方法：

>(__bridge <#type#>)<#expression#>
(__bridge_retained <#CF type#>)<#expression#>
(__bridge_transfer <#Objective-C type#>)<#expression#>)

##桥接方法的用途：

__bridge ：用作于普通的 C 指针与 OC 指针的转换，不做任何操作。

```
void *p;
NSObject *objc = [[NSObject alloc] init];
p = (__bridge void*)objc;
```

这里的 void *p 指针直接指向了 NSObject *objc 这个 OC 类，p 指针并不拥有 OC 对象，跟普通的指针指向地址无疑。所以这个出现了一个问题，OC 对象被释放，p 指针也就 Gameover 了。

__bridge_retained：用作 C 指针与 OC 指针的转换，并且也用拥有着被转换对象的所有权

那么这个是什么意思呢？可以先看下面展示代码

```
@interface ABSClass : NSObject
@property (nonatomic, copy) NSString *name;
@end
@implementation ABSClass
@end
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        void *p;
        {
            ABSClass *objc = [[ABSClass alloc]init];
            objc.name = @"我们";
            p = (__bridge void*)objc;
        }
        NSLog(@"%@", [(__bridge ABSClass *)p name]);
    }
    return 0;
}
```

这段代码看上去大体与上面一段一样，但是我们添加了一个作用域 {} ， 在作用域中创建 ABSClass *objc 这个对象，然后用作用域外的 p，指针进行桥接(__bridge)指向，然后输出 `ABSClass objc这个对象的name属性的值，按道理来说我们会看到控制台上输出我们这两个字。 但是，当我们一运行程序，毫无疑问，程序很崩溃在NSLog(@”%@”, [(__bridge ABSClass )p name]);这句代码中。 有点基础的小伙伴都知道，当ABSClass objc这个对象出了作用域范围，内存就会被回收，但是我们在作用域范围外还用void p去访问objc` 的内存，当然会崩溃啦。
那么，我们尝试修改为以下代码

```
@interface ABSClass : NSObject
@property (nonatomic, copy) NSString *name;
@end
@implementation ABSClass
@end
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        void *p;
        {
            ABSClass *objc = [[ABSClass alloc]init];
            objc.name = @"我们";
            p = (__bridge_retained void*)objc;
        }
        NSLog(@"%@", [(__bridge ABSClass *)p name]);
    }
    return 0;
}
```

程序正常运行，因为我们使用了 __bridge_retained 就相当于 MRC 下的 retain ，将内存计数器 +1，然后用 void *p 指向改内存，所以当 *objc过了作用域，引用计算器 -1，也并没有释放 void *p 所引用的内存。

__bridge_transfer：用作 C 指针与 OC 指针的转换，并在拥有对象所有权后将原先对象所有权释放。(只支持 C 指针转换 OC 对象指针)

说起来相当绕口，其实可以理解为先将对象的引用计数器 +1，然后再将引用计数器 -1。
通过以下代码展现：

```
@interface ABSClass : NSObject
@property (nonatomic, copy) NSString *name;
@end
@implementation ABSClass
@end
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        void *p;
        @autoreleasepool {
            ABSClass *obj = [[ABSClass alloc] init];
            obj.name = @"我们";
            p = (__bridge_retained void *)obj;
        }
        id obj = (__bridge_transfer id)p;
        NSLog(@"%@", [(__bridge ABSClass *)p name]);
        NSLog(@"%@", [(ABSClass *)obj name]);
        NSLog(@"Hello, World!");
    }
    return 0;
}
```

以上代码可以正确运行，在我们将 void *p 指针转换为进行 __bridge_transfer 为 OC 指针，这个操作其实相当于 - (void)set: 操作，转换为 MRC 为如下代码 :

```
id obj = (id)p
[obj retain];
[(id)p release];
```

我们先将新值 retain，然后再将旧值 release，这样是为了保证引用计数器始终为1，一个 retain 对应一个 release。

好了，以上做法就是 C/C++ 指针与 OC 对象指针的相互转换介绍，希望能帮助更多的小伙伴理解。
