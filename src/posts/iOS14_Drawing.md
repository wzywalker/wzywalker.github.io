---
layout: post
title: Programming iOS 14 - Drawing
slug: Drawing
date: 2021-12-25 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - drawing
  - imageview
  - core image
  - path
  - shape
  - graphics
---

《Programming iOS 14: Dive Deep into Views, View Controllers, and Frameworks》第2章

--------
# Drawing

Many UIView subclasses, such as a UIButton or a UILabel, know how to draw themselves. 

A *pure UIView* is all about drawing, and it leaves that drawing largely up to you.

## Images and Image Views

图片可以来自文件，代码，或网络。

### Image Files
* `init(named:)`，会从`Asset catalog`和`App bundle`的顶层去查找
    * 返回的是一个Optional，因为不能确定这个路径对应一张图片，或能解码成功
    * 它会将图片缓存
        * `init(contentsOfFile:)`则不会缓存，但不从asset catalog加载而是相对于`Bundle.main`来做路径
* 从bundle里找时不加扩展名会默认为*png*
* 直接将图片拖到代码生成的不是Optional的image，调用的是`init(imageLiteralResourceName:)`方法
* 文件名里的@表示`High-resolution variants`，即不同分辨率下采用的图片，比如`@2x`
* 文件名里的~表示`Device type variants`，即不同设备类型下采用的图片，比如`~ipad`

> 尽量把图片放到asset catalog里，对不同的处理器，更宽的色域，等等
> 不光影响运行时，在Apple Store对你的app对特定设备进行*thinning*都会用到
> 不同size class, dark mode, ipad等等trait collection都可以设置对应的图片

**Vector images**

* An image file in the asset catalog can be a vector-based PDF or (new in Xcode 12) an SVG.
* `init(systemName:)` -> [SF Symbols](https://developer.apple.com/sf-symbols/)
    * `.withConfiguration(_:) or .applyingSymbolConfiguration(_:)` 进行自定义，参数是一个*UIImage.SymbolConfiguration*
    * Configurations can involve one of nine `weights`, one of three `scales`, a font or text `style`, and a `point size`, in various combinations


**Asset catalogs and trait collections**

指定trait collection初始化图片：`init(named:in:compatibleWith:)`
* A built-in interface object that displays an image, such as a UIImageView, is `automatically trait collection–aware`; 
* it receives the `traitCollectionDidChange(_:)` message and responds accordingly. 

```swift
let tcreg = UITraitCollection(verticalSizeClass: .regular)
let tccom = UITraitCollection(verticalSizeClass: .compact)
let moods = UIImageAsset()
let frowney = UIImage(named:"frowney")!
let smiley = UIImage(named:"smiley")!
moods.register(frowney, with: tcreg)
moods.register(smiley, with: tccom)
```
由此也可见，你操作的是“一张图片”，其实它是一**组**带了条件的图片。

> UIColor也是相同的机制，你用`resolvedColor(with:)`传入trait collection把对应的颜色取出来使用。

**Namespacing image files**

* 物理文件夹，虚拟文件夹内的图片访问时，都需要加上文件夹名（namespaing)
* `init(named:)`的完全形态其实是`init(named:in:)`，第二个参数是bundle，比如来自某个framework.

### Image Views

A UIImageView can actually have two images, one assigned to its `image` property and the other assigned to its `highlightedImage` property
A UIImageView without an image and without a background color is *invisible*

**Resizable Images**

用inset来设置**不**拉伸的区域，比如一般我们碰到的多为左右随便拉伸的胶囊按钮，需要设计师做的就是左右两个半圆（不拉伸）和中间1像素的可拉伸部分
```swift
let marsTiled = mars.resizableImage(withCapInsets:
UIEdgeInsets(
    top: mars.size.height / 2.0 - 1,
    left: mars.size.width / 2.0 - 1,
    bottom: mars.size.height / 2.0 - 1,
    right: mars.size.width / 2.0 - 1
), resizingMode: .stretch)
```
所以如果只是横向拉伸，上面的代码中，top, bottom都可以设为0，或都设为图片高度（而不去除2什么的），只需要保证把UI控件的高度保持跟图片一致即可。

那么，如果不小心高度大于图片高度了呢？分两种情况，如果设了0，表示没有保留区域，直接竖向拉伸，而如果设成了图片高度，那么表示整个Y方向没有可供拉伸的像素，必然造成拉伸失败：

![](../assets/1859625-bf9eea93ad03183f.png)

**Transparency Masks**

The image shown on the screen is formed by combining the image’s `transparency` values with a single `tint color`.

忽略图片各像素上颜色的数值，只保留透明度，就成了一个mask. (renderingMode: `alwaysTemplate`)

* iOS gives every UIView a `tintColor`, which will be used to `tint any template images`。所以我们经常用的tintColor其实就是给模板图片染色的意思。
* tintColor是向下继承的
* The symbol images are always template images
* iOS 13起，可以对UIImage直接应用tint color

**Reversible Images**

* 用`imageFlippedForRightToLeftLayoutDirection`来创建一个在从右向左的书写系统里会自动翻转的图片。
    * 但你又可以设置`semanticContentAttribute`来阻止这个镜像行为
* 如果不考虑书写系统，可以用`withHorizontallyFlippedOrientation`强行镜像

## Graphics Contexts

Graphics Contexts是绘图的起点，你能从如下方式得到Graphics Contexts：
1. 进入UIView的 `draw(_:)`方法时，系统会给你提供一个Graphics Contexts
2. CALayer的`draw(in:)`,或其代理的`draw(_:in:)`方法，*in*参数就是Graphics Contexts
    * 但它不是`currnet context`
3. 手动创建一个

UIKit 和 Core Graphics是两套绘制工具。

* UIKit是大多数情况下你的选择，大部分Cocoa class知道如何绘制自己
* 只能在current context上绘制
* Core Graphics is the full drawing API， often referred to as `Quartz (2D)`
* UIKit drawing is built on top of it.

两套体系，三种context来源，共计6种殊途同归的方式。

### Drawing on Demand

直接上代码：

```swift
// UIView

// UIKit
override func draw(_ rect: CGRect) {
    // 直接绘制
    let p = UIBezierPath(ovalIn: CGRect(0,0,100,100))
    UIColor.blue.setFill()
    p.fill()
}

// CG
override func draw(_ rect: CGRect) {
    // 取到context
    let con = UIGraphicsGetCurrentContext()!
    con.addEllipse(in:CGRect(0,0,100,100))
    con.setFillColor(UIColor.blue.cgColor)
    con.fillPath()
}

// CALayer

// UIKit
 override func draw(_ layer: CALayer, in con: CGContext) {
    UIGraphicsPushContext(con)
    let p = UIBezierPath(ovalIn: CGRect(0,0,100,100))
    UIColor.blue.setFill()
p.fill()
    UIGraphicsPopContext()
}

// CG
override func draw(_ layer: CALayer, in con: CGContext) {
    con.addEllipse(in:CGRect(0,0,100,100))
    con.setFillColor(UIColor.blue.cgColor)
    con.fillPath()
}
```

### Drawing a UIImage

```swift
let r = UIGraphicsImageRenderer(size:CGSize(100,100))
let im = r.image { _ in
    let p = UIBezierPath(ovalIn: CGRect(0,0,100,100))
    UIColor.blue.setFill()
    p.fill()
}
// im is the blue circle image, do something with it here ...
And here’s the same thing using Core Graphics:
let r = UIGraphicsImageRenderer(size:CGSize(100,100))
let im = r.image { _ in
    let con = UIGraphicsGetCurrentContext()!
    con.addEllipse(in:CGRect(0,0,100,100))
    con.setFillColor(UIColor.blue.cgColor)
    con.fillPath()
}
// im is the blue circle image, do something with it here ...
```

## UIImage Drawing

用已有的图像进行绘制：
```swift
let mars = UIImage(named:"Mars")!
let sz = mars.size
let r = UIGraphicsImageRenderer(size:CGSize(sz.width*2, sz.height),
    format:mars.imageRendererFormat)
let im = r.image { _ in
    mars.draw(at:CGPoint(0,0))
    mars.draw(at:CGPoint(sz.width,0))
}
```
这里，绘制了两个火星，注意`imageRendererFormat`的使用

## CGImage Drawing

```swift
let mars = UIImage(named:"Mars")!
// extract each half as CGImage
let marsCG = mars.cgImage!
let sz = mars.size
let marsLeft = marsCG.cropping(to:
    CGRect(0,0,sz.width/2.0,sz.height))!
let marsRight = marsCG.cropping(to:
    CGRect(sz.width/2.0,0,sz.width/2.0,sz.height))!
let r = UIGraphicsImageRenderer(size: CGSize(sz.width*1.5, sz.height),
    format:mars.imageRendererFormat)
let im = r.image { ctx in
    let con = ctx.cgContext
    con.draw(marsLeft, in:
        CGRect(0,0,sz.width/2.0,sz.height))
    con.draw(marsRight, in:
        CGRect(sz.width,0,sz.width/2.0,sz.height))
}
```
当然, `con.draw`可以由UIImage来完成：
```swift
UIImage(cgImage: marsLeft!,
scale: mars.scale,
orientation: mars.imageOrientation).draw(at:CGPoint(0,0))
```

## Snapshots

* `drawHierarchy(in:afterScreenUpdates:)`将整个视图存成一张图片。
* 更快，语义更好的方法：`.snapshotView(afterScreenUpdates:)` -> 输出是UIView，不是UIImage
* `resizableSnapshotView(from:after- ScreenUpdates:withCapInsets:)`生成可缩放的

## Core Image

The “CI” in `CIFilter` and `CIImage` stands for `Core Image`, a technology for transforming images through *mathematical* filters. (iOS 5起，从macOS引入)

用途：
* patterns and gradients (可以被别的filter一起使用)
* compositing (使用composting blend modes)
* color (颜色调整，亮度锐度色温等等)
* geometric (几何相关的就是用来变形)
* transformation (distort, blur, stylize an image)
* transition (一般用于动画，通过设置frame序列)

There are more than 200 available `CIFilters`， A CIFilter is a set of **instructions** for `generating` a CIImage
* 基本上，处理的都是`CIImage`(input)
* 输出也是`CIImage`，或者另一个`filter` -> 链式调用
    * 最后一层链可以自行转换为bitmap: cg或ui image(by `rendering`方法)
    * rendering的时候，所有的数学计算才开始发生
    * 因为只是**instructions**
* **关键词**：filter是用来描述怎么**生成**CIImage的
* `CGImage`和`UIImage`都能得到CIImage

> UIImage只有在已经wraps了一个`CIImage`的情况下`.ciImage`才有值，而大多数情况下是没有的。

[Core Image Filter Reference](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Reference/CoreImageFilterReference/index.html)里有所有的filter的名字，用来初始化一个filter
```swift
let filter = CIFilter(name: "CICheckerboardGenerator")!
// or:
let filter = CIFilter.checkerboardGenerator()

// 用key-value来决定行为：
filter.setValue(30, forKey: "inputWidth")
// or:
filter.width = 30
// or init with params
init(name:parameters:)

// apply filter on CIImage(if exists one)
ciimage.applyingFilter(_:parameters:)
// or output a ciimage
filter.outputImage
```

**Render a CIImage**
CIImage 不是一个`displayaable image`
* `CIContext`.init(options:).createCGImage(_:from)
    * 参数1是CIImage，
    * 参数2是绘制区域（所以没有frame/bounds)，叫`extent`
    * 这是很昂贵的操作，建议在全app生命周期保留这个context复用
* `UIImage`.init(ciImage:)
* 把上一次的uiimage设置成`UIImageView`的image，也能造成CIImage的渲染。

以上说的都是"render" CIImage的时机，所以传入的

> `Metal`能快速渲染CIImage

串起一个demo:
```swift
let moi = UIImage(named:"Moi")!
let moici = CIImage(image:moi)!
let moiextent = moici.extent
let smaller = min(moiextent.width, moiextent.height)
let larger = max(moiextent.width, moiextent.height)
// first filter
let grad = CIFilter.radialGradient()
grad.center = moiextent.center
grad.radius0 = Float(smaller)/2.0 * 0.7
grad.radius1 = Float(larger)/2.0
let gradimage = grad.outputImage!
// 到此步为止，并没有moi这个图片参与，等于是一个纯filter

// second filter
let blend = CIFilter.blendWithMask()
blend.inputImage = moici  // 设置了image
blend.maskImage = gradimage // 这里演示的是mask filter，按我理解并不是链式的，而且语法上也不是链式的，而是赋值给了maskImage，但书里直接说是链式的
let blendimage = blend.outputImage!

// 两种render方法
// content
let moicg = self.context.createCGImage(blendimage, from: moiextent)! // *
self.iv.image = UIImage(cgImage: moicg)

// UIImage
let r = UIGraphicsImageRenderer(size:moiextent.size)
self.iv.image = r.image { _ in
    UIImage(ciImage: blendimage).draw(in:moiextent) // *
}
```

> 关于上述代码里我的疑惑，第一个filter并不是chain到第二个filter里的，但书里说是`obtain the final CIImage in the chain (blendimage)，看来所谓的chain，并不是fitler的chain，而是`outputImage`的chain?
> 问题是，这是唯一且标准的filter嵌套用法么？-> mask

不是的
1. 对filter的outputImage继续应用`aplyingFilter(_:parameters)`来链式应用一个新的filter
    * 返回值是CIImage，不再是filter
    * 所以如果继续chain，直接用返回值调apply...方法即可
2. 把上一个filter的outputImage设为下一个filter的inputImage:

```objective-c
CIFilter *gloom = [CIFilter filterWithName:@"CIGloom"];
[gloom setDefaults];                                        
[gloom setValue: result forKey: kCIInputImageKey];
[gloom setValue: @25.0f forKey: kCIInputRadiusKey];         
[gloom setValue: @0.75f forKey: kCIInputIntensityKey];      
// 即outputImage
CIImage *result = [gloom valueForKey: kCIOutputImageKey];   

CIFilter *bumpDistortion = [CIFilter filterWithName:@"CIBumpDistortion"];
[bumpDistortion setDefaults];                                              
// 设置inputImage (with first filter's output image) 
[bumpDistortion setValue: result forKey: kCIInputImageKey];
[bumpDistortion setValue: [CIVector vectorWithX:200 Y:150]
                forKey: kCIInputCenterKey];                              
[bumpDistortion setValue: @100.0f forKey: kCIInputRadiusKey];                
[bumpDistortion setValue: @3.0f forKey: kCIInputScaleKey];                   
result = [bumpDistortion valueForKey: kCIOutputImageKey];
```

> CIImage能认出EXIF里关于旋转方向的参数，并以正确的方向展示

## Blur and Vibrancy Views

毛玻璃效果，用`UIVisualEffectView`，这是个抽像类，实际用这两个：`UIVisualEffectView`和`UIVibrancyEffect`。

什么是`UIVibrancyEffect`?
>An object that amplifies and adjusts the color of the content layered `behind` a visual effect view.

关键词是`behind`，即它是配合别的视效一起用的（比如毛玻璃）。文字被毛玻璃覆盖后的效果，并不是由毛玻璃层来确定的，而是由vibrancy effect自定义的。

总的来说 
* 用effect初始化effect view, effect就是五种`material`
* 这个view可以当成常规view来定位，布局，添加到subview里，等等
* 用上一个effect初始化一个vibrancy effect（with style)
* 用vibrance effect初始化一个view
* 创建UI控件
* 让vibView的bounds等于内容的bounds（等于只对内容所有的范围内应用特效），并定位
* vibView添加到effectView的contentView的subView里去
* 需要被vibrancy的内容（比如一个label)，则添加到vibView.contentView.addSubview(label)

```swift
let blurEffect = UIBlurEffect(style: .systemThinMaterial)
let blurView = UIVisualEffectView(effect: blurEffect)
blurView.frame = self.view.bounds
blurView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
self.view.addSubview(blurView)
let vibEffect = UIVibrancyEffect(
    blurEffect: blurEffect, style: .label)
let vibView = UIVisualEffectView(effect:vibEffect)
let lab = UILabel()
lab.text = "Hello, world"
lab.sizeToFit()
vibView.bounds = lab.bounds
vibView.center = self.view.bounds.center
vibView.autoresizingMask =
    [.flexibleTopMargin, .flexibleBottomMargin,
    .flexibleLeftMargin, .flexibleRightMargin]
blurView.contentView.addSubview(vibView)
vibView.contentView.addSubview(lab)
```

## Drawing a UIView

UIView本身就提供了一个`graphics context`，在这个context里进行的绘制会直接显示在view里。 
* subclass UIView's `.draw(_:)`method
    * 直到需要时才会被调用
    * 或`setNeedsDisplay`会调用
    * 一量被draw，就缓存起来了 (`bitmap backing store`)
* 实时绘制会吓到一些初学者，绘制是`time-comsuming operation`

推荐在`draw`方法里实时绘制
> In fact, moving code to draw(_: ) is commonly a way to increase efficiency. This is because it is more efficient for the drawing engine to *render directly onto the screen* than for it to *render offscreen* and then copy those pixels onto the screen.

几个注意点：
1. 不要手动调用draw方法，`setNeedsDisplay`会让系统决定下一个合适的时机来draw
2. 不要重载draw方法，比如你无法合并UIImageView的drawing
3. 不要在draw里做任何与绘制无关的事，配置（如背景色，添加子view/layer）项应该在别的地方做，比如`layoutSubviews`
4. 第二个参数是一个rect，默认是view的bounds
    * 如果你用`setNeesDisplay(_:)`送入了自定义的CGRect，draw里面的rect也就成了这个，如果你不在这个rect里画（而是在整个view的rect里），超出部分会被clip掉
    * 这也是为了效率，显示提供绘制的区域
5. 手写draw绘制出来的view会有黑色的底色，如果你没有设计背景色，以及`isOpaque == true`时（`UIView.init(frame:)`出来的view恰好满足这两个条件， nib里拖出来的则是nil的背景，反而没这问题）
    * 解决：实现`init(frame:)`，去设置*isOpaque`为false

## Graphics Context Commands

> Under the hood, Core Graphics commands to a `graphics context` are global C functions with names like CGContextSetFillColor，但是swift的封装让调用更简单（语法糖）

当你在graphics context里绘制时，取的就是当前的设置，因此在任何绘制前，第一步都是先配置context's setting，比如你要画一根红线，再画一根蓝线
1. 设置context line color red, then draw a line
2. 设置context line color blue, then draw a line
直觉认为红和蓝只是两条线各自的属性，其实是你绘制**当时**，整个graphics context的设置
* 这些配置通通存成一个state
* 这些state又会stack起来
    * saveGState将当前state推到栈顶
    * restoreGstate则将state从栈顶取出，覆盖当前设置
* 只要先后配置没有冲突的项，就没必要频繁save-restore

### Paths and Shapes

* 通过一系列的描述去移动一去想象中的笔，就是构建`path`的过程。（注意，不是构建`CGPath`这个封装的过程）
    * 即只要你在context内，就可以用笔画东西了
* 只要你正确地使用`move(to:)`方法，就不需要像apple文档里动不动就用`beginPath`来设置新的path的起点
* `fillPath`会自动`closePaht`
* 先提供path，再draw，draw的意思要么是stroke，要么是fill，要么是both（`drawPath`方法），但不能一步步来，因为draw完你的path就空了
    * 衔接第一条，如果你想复用这个path，才需要用`CGPath`封装起来

* 如果是使用UIKit封装的语法，那么起点就是一个path `let path = UIBezierPath()`
* 那么每次draw完，要在别的位置“落笔”的话，要先清一下靠前的path: `path.removeAllPoints()`

### Clipping

* clipping掉的区域就不能被绘制了
* 通常你无法得知一个graphics context的大小，但是通过`boundingBoxOfClipPath`却能拿到整个bounding

这一节做了几个实验，单独写到了[另一篇博文](https://www.jianshu.com/p/ade133568ac0)

> 前面说过，没有背景色+isOpaque会导致背景变黑，在draw里面，默认的颜色也是黑色，所以你不带任何设置的绘制你是看不到任何东西的（就是黑笔在黑纸上画）

### Gradients

gradient不能用作path的fill，但可以反过来让gradient沿着path分布，以及被clip等。

在上面应用clip绘制箭尾的例子里，我们把箭柄变成从左到右是灰-黑-灰的渐变，只需要在`addLine`并设置了line的宽度后(不要设颜色了），不是去`strokePath()`，而是：
```swift
con.replacePathWithStrokedPath()  // 不再strokePath
con.clip()                        // 再clip一次，奇偶反转
// draw the gradient
let locs : [CGFloat] = [ 0.0, 0.5, 1.0 ]
let colors : [CGFloat] = [
        0.8, 0.4, // starting color, transparent light gray
        0.1, 0.5, // intermediate color, darker less transparent gray
        0.8, 0.4, // ending color, transparent light gray
    ]
let sp = CGColorSpaceCreateDeviceGray()
let grad = CGGradient(
    colorSpace:sp, colorComponents: colors, locations: locs, count: 3)!
con.drawLinearGradient(grad,
    start: CGPoint(89,0), end: CGPoint(111,0), options:[])
con.resetClip() // done clipping
```

小技巧就是用`replacePathWithStrokedPath`假装进行了描边（所以只需要线宽并不需要线的颜色），返回了一个新的path，一条粗线变成了一个矩形框。   
而一旦添加了这个框，前面的奇偶关系就全反过来了，于是我们再`clip`一次，这就是头两行代码里做的事。

### Colors and Patterns

当你的suer interface sytle changes(比如黑暗模式切换), `draw(_:)`方法会被立刻调用，被设置`UITraitCollection.current`，任何支持动态颜色的`UIColor`能变成相应的颜色，但是`CGColor`不能，你需要手动触发重绘。

UIKit使用pattern非常简单，把纹理绘制到图片上，然后从纹理图片提取出颜色信息，就能像别的颜色一样`setFill`了：
```swift
// create the pattern image tile
let r = UIGraphicsImageRenderer(size:CGSize(4,4))
let stripes = r.image { ctx in
    let imcon = ctx.cgContext
    imcon.setFillColor(UIColor.red.cgColor)
    imcon.fill(CGRect(0,0,4,4))
    imcon.setFillColor(UIColor.blue.cgColor)
    imcon.fill(CGRect(0,0,4,2))
}
// paint the point of the arrow with it
let stripesPattern = UIColor(patternImage:stripes)
stripesPattern.setFill()
let p = UIBezierPath()
p.move(to:CGPoint(80,25))
p.addLine(to:CGPoint(100,0))
p.addLine(to:CGPoint(120,25))
p.fill()
```

而Core Graphics则要复杂（也更底层）得多，结合注释看代码：
```swift
con.saveGState()
// 非常重要，设置颜色空间
let sp2 = CGColorSpace(patternBaseSpace:nil)!
con.setFillColorSpace(sp2)
// 纹理绘制真正发生的地方
let drawStripes : CGPatternDrawPatternCallback = { _, con in
    con.setFillColor(UIColor.red.cgColor)
    con.fill(CGRect(0,0,4,4))
    con.setFillColor(UIColor.blue.cgColor)
    con.fill(CGRect(0,0,4,2))
}
// 包装成一个callback给CGPattern使用
var callbacks = CGPatternCallbacks(
    version: 0, drawPattern: drawStripes, releaseInfo: nil) // 一个struct

// 核心就是构造这个CGPattern
let patt = CGPattern(info:nil, bounds: CGRect(0,0,4,4),  // cell大小
    matrix: .identity,    // cell变换，这里没有，就用.identity
    xStep: 4, yStep: 4,   // 横向纵向复制cell时的步长
    tiling: .constantSpacingMinimalDistortion,  // 排列方式
    isColored: true,      // 是颜色还是画笔模式，选颜色true
    callbacks: &callbacks)!  // 纹理绘制的方法包在callback里面，传指针
var alph : CGFloat = 1.0
con.setFillPattern(patt, colorComponents: &alph)
con.move(to:CGPoint(80, 25))
con.addLine(to:CGPoint(100, 0))
con.addLine(to:CGPoint(120, 25))
con.fillPath()
con.restoreGState()
```

### Graphics Context Transforms

跟前面的知识点一样，应用*Graphics Context Transforms*后，也不会影响当前已经绘制的东西。 => `CTM`即（`current transform matrix`)。

旋转的中心点是原点，大多数情况下不是你想要的，记得先translate一下。

```swift
override func draw(_ rect: CGRect) {
    let con = UIGraphicsGetCurrentContext()!
    con.setShadow(offset: CGSize(7, 7), blur: 12) // 顺便演示下sahdow
    con.beginTransparencyLayer(auxiliaryInfo: nil)  // 这样重叠的阴影不会叠成黑色
    self.arrow.draw(at:CGPoint(0,0))
    for _ in 0..<3 {
        con.translateBy(x: 20, y: 100)
        con.rotate(by: 30 * .pi/180.0)
        con.translateBy(x: -20, y: -100)
        self.arrow.draw(at:CGPoint(0,0)) // 注意这里是用前面方法生成的箭头图片来draw到指定位置
    } 
}
```

![](../assets/1859625-e406c90a2896eeaf.png)

注意，语法虽然是先处理context，再绘制，其实只是告知坐标系的变化，绘制的时候自动应用这些变换。

### Erasing

`clear(_:)`擦除行为取决于context是透明还是实心的（透明擦成透明，实心擦成黑色），只要不是opaque，通通理解为透明，比如background color是nil, 或0.9999的透明度。

## Points and Pixels

`con.fill(CGRect(100,0,1.0/self.contentScaleFactor,100))`应用contentScaleFactor画一条在任何屏幕上都锐利的1像素直线。

## Content Mode

the drawing system will `avoid` asking a view to `redraw` itself from scratch if possible; instead, it will use the `cached` result of the previous drawing operation (the **bitmap backing store**). 

If the view is resized, the system may simply stretch or shrink or reposition the cached drawing, if your contentMode setting instructs it to do so.


`draw(_:)`从原点开始绘制，所以你的`contentMode`也要相应设置为`topLeft`。而如果设置为`.redraw`，则不会使用cached content，每当view被resize的时候，就会调用`setNeedsDisplay`方法，最终触发`draw(_:)`进行重绘。
