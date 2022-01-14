---
layout: post
title: cs193p_2021笔记[7]_Document-Architecture
slug: cs193p_2021笔记[7]_Document-Architecture
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
[cs193p_2021_笔记_5_Property Wrapper](https://www.jianshu.com/p/e3c2ee1628c6)
[cs193p_2021_笔记_6_Persistence](https://www.jianshu.com/p/a315274a4fd2)
cs193p_2021_笔记_7_Document Architecture
[cs193p_2021_笔记_8](https://www.jianshu.com/p/2136bdc2c6f6)

# Document Architecture

所谓的Dopcument Architecture，其实就是支持把用app产生的作品保存起来，比如你创作的一幅图片，可以保存为`.jpg`，你用photoshop做的文件是`.psd`，下次用自己的app加载这个文件，能认出所有组件和模型，比如我们想为document取个名字叫`.emojiart`。

## App Architecture

### App protocol

* 一个app里只能有一个struct服从`App Protocol`
* mark it with `@main`
* it's `var body` is `some Scene`

### Scene protocol

* A `Scene` is a container fo a `top-lever` View that you want to show in your UI
* `@Environment(\.scenePhase)`
* three main types of Scenes:
```swift
WindowGroup {return aTopLevelView}
DocumentGroup(newDocument:) { config in ... return aTopLevelView}
DocumentGroup(viewing: viewer:) { config in ... return aTopLevelView}  // 只读
```
* 后两个类似view里面的`ForEach`但不完全相同：
    * 而是："**new window**" on Mac, "**splitting the screen**" on iPad -> for create new Scene
* `content`参数是一个返回some View的方法
    * 返回的是top-level view
    * 每当新建一个窗口或窗口被分割时都会被调用

当你在iPad上分屏，且两个打开同一应用，就是`WindowGroup`在管理，为每一个windows生成一个Scene(share the same parameter e.g. view model, 因为代码是同一份，除非额外为每个scene设置自己的viewmodel之类的).

`config`里保存了document(即viewModel)，也保存了文件位置。

### SceneStorage

* 能持久化数据
* 以窗口/分屏为单位 -> per-Scene basis
* 也会invalidate view
* 数据类型有严格限制，最通用的是`RawRepresentable`

[图片上传失败...(image-66d359-1636448439942)]

一个View里的`@State`改为`@SceneStorage(uniq_id)`后，app退出或crash了，仍然能找回原来的值。

这个时候每个Scene里的值就已经不一样了。

### AppStorage

* application-wide basis
* 存在UserDefaults里
* 服从`@SceneStorage`的数据才能被存储
* invalidate view

## DocumentGroup

`DocumentGroup` is the document-oriented Scene-building Scene.

```swift
@main
struct MyDemoApp: App {
    @StateObject var paletteStore = PaletteStore(named: "Default")
    var body: some Scene {
        WindowGroup {
            MyDemoView()
            .environmentObject(paletteStore)
        }
    }
}

// V.S.

struct MyDemoApp: App {
    var body: some Scene {
        DocumentGroup(newDocument: {myDocument()}) { config in
            MyDemoView(document: config.document)
        }
    }
}
```

* 不再用`@StateObject`传递ViewModel，每新建一个Document都会有一个独立的ViewModel
    * 必须要服从`ReferenceFileDocument`(这样能存到文件系统以及从文件系统读取了)
    * `config`参数包含了这个ViewModel（就是document)，以及document的url
    * 很好理解，每一个document肯定有自己的数据（想象一个“最近打开”的功能，每一个文档都是独立的）
* `newDocument`里自行提供一个新建document的方法
* 封装了关联的（选择document的）UI和行为
* you **MUST** implement `Undo` in your application

如果不去实现`Undo`，也可以直接把model存到document文件里：
1. 你的ViewModel要能init itself from a `Binding<Type>`
    * 如`config.$document`
2. ViewModel由一个`ObservedObject`变成一个`StateObject`
    * 这次必须服从`FileDocument`

```swift
struct MyDemoApp: App {
    var body: some Scene {
        DocumentGroup(newDocument: {myDocument()}) { config in
            // MyDemoView(document: config.document) // 之前的
            MyDemoView(document: viewModel(model: config.$document))
        }
    }
}
```

把`newDocument: {myDocument()}`改为`viewer: myDocument.self`，就成了一个只读的model，（你甚至不需要传入实例），如果你要开发的是一个查看别人文档的应用，这个特性就比较有用了。

### FileDocument protocol

This protocol gets/puts the contents of a document from/to a file. 即提供你的document读到文件系统的能力。

```swift
// create from a file
init(configuration: ReadConfiguration) throws {
    if let data = configuration.file.regularFileContents {
        // init yourself from data
    } else {
        throw CocoaError(.fileReadCorruptFile)
    }
}

// write
func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
    FileWrapper(regularFileWithContents: /*my data*/)
}
```

### ReferenceFileDocument

* 几乎和`FileDocument`一致
* 继承自`ObservableObject` -> ViewModel only
* 唯一的区别是通过后台线程的一个`snapshot`来写入
```swift
// 先snapshot
func snapshot(contentType: UTType) throws -> Snapshot {
    return // my data or something
}
// then write
func fileWrapper(snapshot: Snapshot, configuration: WriteConfiguration) throws -> FileWrapper {
    FileWrapper(regularFileWithContents: /* snapshpt converted to a Data */)
}
```

流程大概是，你的model有变化之后，会先找`snapshot`方法创建一份镜像，然后再要求你给出一个`fileWrapper`来写文件。

### 自定义文件类型

声明能打开什么类型的文件，通过：UTType(`Uniform Type Identifier`)

可以理解为怎么定义并注册（关联）自己的扩展名，就像photoshop关联.psd一样。

1. 声明(Info tab)，设置`Exported/Imported Type Identifier`，所以表面上的扩展名，内里还对应了一个唯一的标识符，一般用反域名的格式
![](../assets/1859625-6647d6e837918894.png)

2. 声明拥有权，用的就是上一步标识符，而不是扩展名
![](../assets/1859625-7f6bd8787f7a52b4.png)

3. 告知系统能在`Files` app里打开这种文档
    * info.plist > Supports Document Browser > YES
4. 代码里添加枚举：
```swift
extension UTType {
    static let emojiart = UTType(exportedAs: "edu.bla.bla.emojimart")
}

static let readableContentTypes = [UTType.emojiart]
```

## Undo

* use `ReferenceFileDocument` must implement Undo
* 这也是SwiftUI能自动保存的时间节点
* by `UndoManager` -> `@Environment(\.undoManager) var undoManager`
* and by register an `Undo` for it: `func registerUndo(withTarget: self, howToUndo: (target) -> Void)`

```swift
func undoablePerform(operation: String, with undoManager: UndoManager?, doit: () -> Void){
    let oldModel = model
    doit()
    undoManager?.registerUndo(withTarget: self) { myself in
        myself.model = model
    }
    undoManager?.setActionName(operation) // 给操作一个名字，如"undo paste"， 非必需
}
```

用`undoablyPerform(with:){} 包住的任何改变model的操作就都支持了undo

## Review

回顾一下，我们把应用改造为`Document Architechture`的步骤：
1. 应用入口，将`WindowGroup`改为了`DocumentGroup`，并修改了相应的传递document的方式
2. 实现document(即view model) comform to `ReferenceFileDocument`
    * 实现snapshot, write to file (`FileWrapper`), and read from file
3. 自定义一个文件类别（扩展名，标识符，声明拥有者等）
4. 此时启动应用，入口UI已经是文档选择界面了，所以我说它封装了UI和行为
    * 但此时不具备保存的功能，需要进一步实现`Undo`'
5. 通过`undoManager`把改动model的行为都包进去实现undo/redo
    * 此时document已能自动保存
6. 增加toolbar, 实现手动undo/redo
7. 顺便注册文档类型，以便在Files应用内能用本app打开
    * `Info.plist` > `Supports Document Browser` > YES
