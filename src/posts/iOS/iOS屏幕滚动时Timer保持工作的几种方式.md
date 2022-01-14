---
layout: post
title: iOS屏幕滚动时Timer保持工作的几种方式
slug: iOS屏幕滚动时Timer保持工作的几种方式
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

iOS当前线程的RunLoop在TableView等scrollView滑动时将DefaultMode切换到了`TrackingRunLoopMode`。因为Timer默认是添加在RunLoop上的`DefaultMode`上的，当Mode切换后Timer就停止了运行。
如这样:
```Objective-C
Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { (timer) in
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "HH:mm:ss"
    self.timeLabel.text = "\(dateFormatter.string(from: Date()))"
}
```
本文记录如下四种方式:
+ 将NSTimer添加到当前线程所对应的RunLoop中的`commonModes`中。
+ 通过Dispatch中的`TimerSource`来实现定时器。
+ 是开启一个新的子线程，将NSTimer添加到这个子线程中的RunLoop中，并使用`DefaultRunLoopModes`来执行。
+ 使用`CADisplayLink`来实现。

## CommonModes
```Objective-C
override func awakeFromNib() {
    super.awakeFromNib()
    
    let timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { (timer) in
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss"
        self.timeLabel.text = "\(dateFormatter.string(from: Date()))"
    }
    
    RunLoop.current.add(timer, forMode: .commonModes)
}
```
## 子线程/异步 + DefaultMode
```Objective-C
override func awakeFromNib() {
    super.awakeFromNib()
    DispatchQueue.global().async {
        let timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { (timer) in
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "HH:mm:ss"
            DispatchQueue.main.async {
                self.timeLabel.text = "\(dateFormatter.string(from: Date()))"
            }
        }
        RunLoop.current.add(timer, forMode: .defaultRunLoopMode)
        RunLoop.current.run()
    }
}
```

## DispatchTimerSource
GCD的知识点
```Objective-C
override func awakeFromNib() {
    let queue: DispatchQueue = DispatchQueue.global()   //也可以用mainQueue来实现
    let source = DispatchSource.makeTimerSource(flags: DispatchSource.TimerFlags(rawValue: 0), queue: queue)
    let timer = UInt64(1) * NSEC_PER_SEC
    
    source.scheduleRepeating(deadline: DispatchTime.init(uptimeNanoseconds: UInt64(timer)), interval: DispatchTimeInterval.seconds(Int(1)), leeway: DispatchTimeInterval.seconds(0))
    
    let timeout = 1
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "HH:mm:ss"
    source.setEventHandler {
        if(timeout < 0) {
            source.cancel()
        }
    
        DispatchQueue.main.async {
            self.timeLabel.text = "\(dateFormatter.string(from: Date()))"
        }
    }
    source.resume()
}
```
## CADisplayLink
CADisplayLink可以添加到RunLoop中，RunLoop的每一次循环都会触发CADisplayLink所关联的方法。在屏幕不卡顿的情况下，每次循环的时间时1/60秒。
```Objective-C
override func awakeFromNib() {
    super.awakeFromNib()
    DispatchQueue.global().async {
        let displayLink = CADisplayLink(target: self, selector: #selector(self.update))
        displayLink.add(to: RunLoop.current, forMode: .defaultRunLoopMode)
        RunLoop.current.run()
    }
}

func update() {
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "HH:mm:ss"
    let time = "\(dateFormatter.string(from: Date()))"
    
    if time != currentTime {
        currentTime = time
        DispatchQueue.main.async {
            self.timeLabel.text = self.currentTime
        }
    }
}
```
详细内容请[阅读原文](https://mp.weixin.qq.com/s/amgKKHhOCJ10Mr-OBEQyjw)
