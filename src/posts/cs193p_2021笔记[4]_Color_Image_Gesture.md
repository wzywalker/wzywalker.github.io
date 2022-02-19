---
layout: post
title: cs193p_2021笔记[4]_Color_Image_Gesture
slug: cs193p_2021笔记[4]_Color_Image_Gesture
date: 2021-10-24 03:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - uicolor
  - uiimage
  - gesture
  - item provider
---

# Color, UIColor & CGColor

Color:
* Is a color-specifier, e.g., `.foregroundColor(Color.green)`.
* Can also act like a `ShapeStyle`, e.g., `.fill(Color.blue)`.
* Can also act like a `View`, e.g., Color.white can appear `wherever` a View can appear.（可以当作view）

UIColor:
* Is used to `manipulate` colors.（主打操控）
* Also has many `more` built-in `colors` than `Color`, including “system-related” colors.(颜色更多)
* Can be interrogated and can convert between color spaces.

For example, you can get the RGBA values from a UIColor.
Once you have desired UIColor, employ `Color(uiColor:)` to use it in one of the roles above.

CGColor:
* The fundamental color representation in the Core Graphics drawing system
* `color.cgColor`

# Image V.S. UIImage

Image:
* Primarily serves as a View.(主要功能是View)
* Is `not` a type for vars that `hold an image` (i.e. a jpeg or gif or some such). That’s UIImage. 
* Access images in your Assets.xcassets (in Xcode) by name using `Image(_ name: String)`. 
* Also, many, many system images available via `Image(systemName:)`.
* You can control the size of system images with `.imageScale()` View modifier.
* System images also are affected by the .font modifier. 
* System images are also very useful `as masks` (for gradients, for example).

UIImage
* Is the type for actually `creating/manipulating` images and `storing` in vars. 
* Very powerful representation of an image.
* Multiple file formats, transformation primitives, animated images, etc. 
* Once you have the UIImage you want, use Image(uiImage:) to display it.

# Multithreading

* 多线程其实并不是同时运行，而是前后台非常快速地切换
* `Queue`只是有顺序执行的代码，封装了`threading`的应用
* 这些“代码”用`closure`来传递
* **main queue**唯一能操作UI的线程
    * 主线程是单线程，所以不能执行异步代码
* **background queues**执行任意：*long-lived, non-UI* tasks
    * 可以并行运行(running in parallel) -> even with main UI queue
    * 可以手动设置优先级，服务质量(`QoS`)等
    * 优先级永远不可能超过main queue
* base API: GCD (`Grand Central Dispatch`)
    1. getting access to a queue
    2. plopping a block of code on a queue

A: Creating a Queue

There are numerous ways to create a queue, but we’re only going to look at two ...
```swift
DispatchQueue.main // the queue where all UI code must be posted
DispatchQueue.global(qos: QoS) // a non-UI queue with a certain quality of service qos (quality of service) is one of the following ...
    .userInteractive    // do this fast, the UI depends on it!
    .userInitiated  // the user just asked to do this, so do it now
    .utility    // this needs to happen, but the user didn’t just ask for it
    .background // maintenance tasks (cleanups, etc.)
```

B: Plopping a Closure onto a Queue

There are two basic ways to add a closure to a queue ...
```swift
let queue = DispatchQueue.main //or
let queue = DispatchQueue.global(qos:) 
queue.async { /* code to execute on queue */ }
queue.sync { /* code to execute on queue */ }
```

主线程里永远不要`.sync`, 那样会阻塞UI

```swift
DispatchQueue(global: .userInitiated).async {
    // 耗时代码
    // 不阻塞UI，也不能更新UI
    // 到主线程去更新UI
    DispatchQueue.main.async {
        // UI code can go here! we’re on the main queue! 
    }
}
```

# Gestures

手势是iOS里的一等公民
```swift
// recognize
myView.gesture(theGesture) // theGesture must implement the Gesture protocol

// create
var theGesture: some Gesture {
    return TapGesture(count: 2)  // double tap
}

// discrete gestures
var theGesture: some Gesture {
      return TapGesture(count: 2)
        .onEnded { /* do something */ }
}

// 其实就是：
func theGesture() -> some Gesture {
    tapGesture(count: 2)
}

// “convenience versions”
myView.onTapGesture(count: Int) { /* do something */ } 
myView.onLongPressGesture(...) { /* do something */ }

// non-discrete gestures

var theGesture: some Gesture {
      DragGesture(...)
.onEnded { value in /* do something */ } 

```

non-discrete手势里传递的`value`是一个state:

* For a `DragGesture`, it’s a struct with things like the `start and end location` of the fingers.
* For a `MagnificationGesture` it’s the `scale` of the magnification (how far the fingers spread out). 
* For a `RotationGesture` it’s the `Angle` of the rotation (like the fingers were turning a dial).
* 还可以跟踪一个state: `@GestureState var myGestureState: MyGestureStateType = <starting value>`

唯一可以更新这个`myGestureState`的机会：
```swift
var theGesture: some Gesture {
     DragGesture(...)
        .updating($myGestureState) { value, myGestureState, transaction in 
            myGestureState = /* usually something related to value */
        }
        .onEnded { value in /* do something */ }
 }
 ```
 注意`$`的用法
 
 如果不需要去计算一个`gestureState`传出去的话，有个`updating`用简版：
 ```swift
 .onChanged { value in
/* do something with value (which is the state of the fingers) */
}
```
事实上，目前来看`gestureState`只做了两件事：
1. 把实时手势对应的值保存起来
2. 在手势结束时复原（对于缩放，变为1，对于移动，变为0）
3. 同时，它是只读的，只在`.updating`方法里有更新的机会

所以，如果你的UI和动画逻辑，用到了手势结束时的值（即需要它复原），那么你也可以直接在`.onEnded`方法里手动把它设回去，等同于你也实现了你的`gestureState`，并且没有它那些限制。

## Drag and Drop

### Item Provider

* The heart of drag nad drop is the `NSItemProvider` class.
* It facilitates the transfer of data between processes (via drag and drop, for example)
* It facilitates the transfer of a number of data types in iOS, for example:
    * NSAttributedString and NSString
    * NSURL
    * UIImage and UIColor
* pre-Swift，所以需要bridging，比如：`String as NSString`

结合几个要点，一句话就能让你的元素能被拖动(drag)：
```swift
Text(emoji).onDrag{ NSItemProvider(object: emoji as NSString)}
```

而接收(drop)则要复杂很多：
```swift
otherView.onDrop(of: [.plainText], isTarget: nil) {providers, location in return false }
```
* 参接收的类型由`of`参数指定，这里假定是文本
* 方法里最终要返回一个bool值，表示成功接收与否，我返了个false，意思是你能让物体拖动，但是一松开手指就复原了

从`itemprovider`里加载对象有模板代码：

```swift
extension Array where Element == NSItemProvider {
  func loadObjects<T>(ofType theType: T.Type, firstOnly: Bool = false, using load: @escaping (T) -> Void) -> Bool where T: NSItemProviderReading {
    if let provider = first(where: { $0.canLoadObject(ofClass: theType)}) {
      provider.loadObject(ofClass: theType) { object, error in
        if let value = object as? T {
          DispatchQueue.main.async {
              load(value)
          }
        }
      }
      return true
    }
    return false
  }

// and
// where T: _ObjectiveCBridgeable, T._ObjectiveCType: NSItemProviderReading
```

1. 提供了两段代码，可以看到其实就是对要加载的对象的约束不同，提供了对OC的兼容
2. 模板代码演示了
稳健地从拖拽对象加载内容（canload -> load)
3. 真正的业务逻辑其实就是为拖进来的这个view选择一个位置存放（或读取它携带的数据）
4. `T.Type`传的是类别的`.self`，比如`String.self`
