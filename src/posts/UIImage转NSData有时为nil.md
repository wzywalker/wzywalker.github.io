---
layout: post
title: UIImage转NSData有时为nil
slug: UIImage转NSData有时为nil
date: 2018-07-14 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - debug
  - uiimage
---

一般, 我们会用`UIImagePNGRepresentation`, `UIImagePNGRepresentation`来达到目的, 但有时候, 发现它的返回值为`nil`...

不需要怀疑这么简单处理有什么问题, [文档](https://developer.apple.com/library/ios/documentation/UIKit/Reference/UIKitFunctionReference/#//apple_ref/c/func/UIImagePNGRepresentation) 就是如此:

> **Return Value**
>
> A data object containing the PNG data, or nil if there was a problem generating the data. **This function may return nil if the image has no data** or if the underlying CGImageRef contains data in an unsupported bitmap format.

也就是说, 没有`data`的情况还是挺多的, 我们还是放弃这个方法吧, 换别的吧, 提供三种思路

## 复制一张图片

```swift
var imageName: String = "MyImageName.png"
var image = UIImage(named: imageName)
var rep = UIImagePNGRepresentation(image)
```

当然, 这不能保证什么

## 重绘一张图片

```objective-c
UIGraphicsBeginImageContext(originalImage.size);
[originalImage drawInRect:CGRectMake(0, 0, originalImage.size.width, originalImage.size.height)];
UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
UIGraphicsEndImageContext();
```

或

```swift
UIGraphicsBeginImageContext(CGSizeMake(1, 1))
let image = UIGraphicsGetImageFromCurrentImageContext()
UIGraphicsEndImageContext()
```

## 不用 UIImage

1和2都没验证过, 但都是在StackOverflow上别人贴出的答案, 我之所以不验证了, 因为我是这么做的

```objective-c
CGDataProviderRef provider = CGImageGetDataProvider(image.CGImage);
NSData* data = (id)CFBridgingRelease(CGDataProviderCopyData(provider));
```

通了就没动力继续试啦, 而且本身已经很简洁了, 此外方法名也非常直白"`DataProvider`", 还想怎样!
