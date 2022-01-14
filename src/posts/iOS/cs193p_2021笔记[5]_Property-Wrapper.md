---
layout: post
title: cs193p_2021笔记[5]_Property-Wrapper
slug: cs193p_2021笔记[5]_Property-Wrapper
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

[cs193p_2021_笔记_1](https://www.jianshu.com/p/998b0ef4a2cd)
[cs193p_2021_笔记_2](https://www.jianshu.com/p/af0ad1bead34)
[cs193p_2021_笔记_3_Animation_Transition](https://www.jianshu.com/p/d103f8d12052)
[cs193p_2021_笔记_4_Color_Image_Gesture](https://www.jianshu.com/p/41e7309c7f55)
cs193p_2021_笔记_5_Property Wrapper
[cs193p_2021_笔记_6_Persistence](https://www.jianshu.com/p/a315274a4fd2)
[cs193p_2021_笔记_7_Document Architecture](https://www.jianshu.com/p/f4ae879eef9c)
[cs193p_2021_笔记_8](https://www.jianshu.com/p/2136bdc2c6f6)

---

#  Property Wrappers

C#中的`Attributes`，python中的`Decorators`, Java的`Annonations`，类似的设计模式。

* A property wrapper is actually a `struct`.
* 这个特殊的`struct`封装了一些模板行为应用到它们wrap的vars上：
    1. Making a var live in the heap (`@State`)
    2. Making a var publish its changes (`@Published`)
    3. Causing a View to redraw when a published change is detected (`@ObservedObject`)

即能够分配到堆上，能够通知状态变化和能重绘等，可以理解为`语法糖`。

```swift
@Published var emojiArt: EmojiArt = EmojiArt()

// ... is really just this struct ...
struct Published {
    var wrappedValue: EmojiArt
    var projectedValue: Publisher<EmojiArt, Never>  // i.e. $
}

// `projected value`的类型取决于wrapper自己，比如本例就是一个`Publisher`

// 我理解为一个属性和一个广播器

// ... and Swift (approximately) makes these vars available to you ...
var _emojiArt: Published = Published(wrappedValue: EmojiArt()) 
var emojiArt: EmojiArt {
     get { _emojiArt.wrappedValue }
     set { _emojiArt.wrappedValue = newValue }
 }
```

把get,set直接通过`$emojiArt`(即projectedValue)来使用

当一个`Published`值发生变化：
* It publishes the change through its *projectedValue* (`$emojiArt`) which is a `Publisher`. 
* It also invokes `objectWillChange.send()` in its enclosing `ObservableObject`.

下面列的几种`Property wrapper`，我们主要关心最核心的两个概念，`wrappedValue`和`projectedValue`是什么就行了:

## @State

这是第二次提到了，在`Property Observers`一节里预告过，基本上点`@`的，大都为`Property Wrapper`的内容。

* The wrappedValue is: `anything` (but almost certainly a value type).
* What it does: 
    * stores the wrappedValue in the heap; 
    * when it changes, `invalidates` the `View`. 
* Projected value (i.e. $): a `Binding` (to that *value in the heap*).

```swift
@State private var foo: Int
init() {
    _foo = .init(initiaValue: 5)
}
```

注意`_`和`$`的区别。 

## @StateObject & @ObservedObject

* The wrappedValue is: `anything` that implements the `ObservableObject` protocol (ViewModels). 
* What it does: 
    * `invalidates` the `View` when wrappedValue does *objectWillChange.send()*. 
* Projected value (i.e. $): a `Binding` (to the vars of the wrappedValue (a *ViewModel*)).

> **@StateObject V.S. @State**

* 一个类型是`ObservableObject`s， 一个是value type

> **@StateObject V.S. @ObservedObject**
 
* @StateObject is a "source of truth"，也就是说可以直接赋值：`@StateObject var foo = SomeObservableObject()`
* 能用在*View, APP, Scene*等场景
* 如果用在View里，生命周期与View一致 

```swift
@main
struct EmojiArtApp: App {
    // stateObject, source of truth
    // defined in the app
    @StateObject var paletteStore = PaletteStore(named: "default")

    var body: some Scene {
    DocumentGroup(newDocument: { EmojiArtDocument() }) { config in
        EmojiArtDocumentView(document: config.document)
            .environmentObject(paletteStore)  // passed by environment
        }
    }
}
```

## @Binding

* The wrappedValue is: `a value` that is bound to something else.
* What it does: 
    * gets/sets the value of the wrappedValue from `some other source`. 
    * when the bound-to value changes, it `invalidates` the `View`. 
    * Form表单典型应用场景，有UI变化的控件
    * 手势过程中的State, 或drag时是否targted
    * 模态窗口的状态
    * 分割view后共享状态
    * 总之，数据源只有一个(source of the truth)的场景，就不需要用两个@State而用@Binding, 
* Projected value (i.e. $): a Binding (self; i.e. the Binding itself)

```swift
struct MyView: View {
      @State var myString = “Hello”               // 1
      var body: View {
          OtherView(sharedText: $myString)        // 2
      }
  }
  struct OtherView: View {
      @Binding var sharedText: string             // 3
      var body: View {
          Text(sharedText)                        // 4
          TextField("shared", text: $sharedText)  // 5 _myString.projectValue.projectValue
      }
}
```
1. `_myString`是实际变量，包含一个`wrappedValue`，一个`projectedValue`
2. `myString`就是`_myString.wrappedValue`
3. `$myString`是`_myString.projectedValue`，
    * 是一个`Binding<String>`，传值和接值用的就是它
    * 所以传`$myString`的地方也可以用`_myString.projectedValue`代替，学习阶段的话
4. 要把`projectedValue`层层传递下去，并不是用同一个`projectedValue`，而是设计成了`Binding<T>`
    * 参考上面代码块的第5条

其它
* 也可以绑定一个常量：`OtherView(sharedText: .constant(“Howdy”))`
* computed binding: `Binding(get:, set:).`

比如你的view是一个小组件，里面有一个`Binding var user: User`，那么在preview里面怎么传入这个User呢？用常量：
```swift
static var preview: some View {
    myView(user: .constant(User(...)))
}
```

## @EnvironmenetObject

* The wrappedValue is: `ObservableObject` obtained via .environmentObject() sent to the View. 
* What it does: `invalidates` the View when wrappedValue does objectWillChange.send(). 
* Projected value (i.e. $): a `Binding` (to the vars of the wrappedValue (a ViewModel)).

与`@ObservedObject`用法稍有点不同，有单独的赋值接口：

```swift
let myView = MyView().environmentObject(theViewModel)
// 而@ObservedObject是一个普通的属性
let myView = MyView(viewModel: theViewModel)

// Inside the View ...
@EnvironmentObject var viewModel: ViewModelClass 
// ... vs ...
@ObservedObject var viewModel: ViewModelClass
```
* visible to all views in your body (except modallay presented ones)
* 多用于多个view共享ViewModel的时候

## @Environment

* 与`@EnvironmentObject`完全不是同一个东西
* 这是`Property Wrapper`不只有两个变量（warped..., projected...）的的一个应用
* 通过`keyPath`来使用：`@Environment(\.colorScheme) var colorScheme`
* wrappedValue的类型是通过`keyPath`声明时设置的

```swift
view.environment(\.colorScheme, .dark)
```

so:

* The wrappedValue is: the value of some var in `EnvironmentValues`. 
* What it does: gets/sets a value of some var in `EnvironmentValues`. 
* Projected value (i.e. $): none.

```swift
// someView pop 一个 modal 的 myView,传递 environment
someView.sheet(isPresented: myCondition){
    myView(...init...)
    .enviroment(\.colorScheme, colorScheme) 
}
```

除了深色模式，还有一个典型的应用场景就是编辑模式`\.editMode`，比如点了编辑按钮后。

> `EditButton`是一个封装了UI和行为的控件，它只做一件事，就是更改`\.editmode`这个环境变量(的`isEditing`)

## @Publisher

It is an object that `emits values` and possibly a `failure object` if it fails while doing so.
```swift
Publisher<Output, Failure>
```
* Failure需要实现`Error`，如果没有，可以传`Never`

### 订阅

一种简单用法，`sink`:
```swift
cancellable = myPublisher.sink(
    receiveCompletion:{resultin...}, //result is a Completion<Failure> enum
        receiveValue: { thingThePublisherPublishes in . . . }
  )
```
返回一个`Cancellable`，可以随时`.cancel()`，只要你持有这个`cancellable`，就能随时用这个sink

View有自己的订阅方式：
```swift
.onReceive(publisher) { thingThePublisherPublishes in
    // do whatever you want with thingThePublisherPublishes 
}
```
1. `.onReceive` will automatically `invalidate` your View (causing a redraw).
2. 既然参数是publisher，所以是一个binding的变量，即带`$`使用：
```swift
.onReceive($aBindData) { bind_data in 
    // my code
}
```

publisher来源：
1. `$` in front of vars marked `@Published`
    * 还记得$就是取的projectedValue吗？
    * 一般的projectedValue是一个*Binding*，Published的是是个*Publisher*
2. URLSession’s `dataTaskPublisher` (publishes the Data obtained from a URL)
3. `Timer`’s publish(every:) (periodically publishes the current date and time as a Date) 
4. `NotificationCenter`’s publisher(for:) (publishes notifications when system events happen)

> 如果你有一个`ObservedObject`(Document)，它里面有一个`@Publisher`(background)，那么注意以下两者的区别：

* document.`$`background: 是一个publisher
* `$`document.background: 是一个binding

> `.onReceive`只能接收`Publisher`的推送，而事实上，`onChange`（一般用于接收ObservedObject或State)同样也能接收Publisher。
