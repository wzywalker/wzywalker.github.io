---
layout: post
title: iOS应用3D-Touch快捷访问
slug: iOS应用3D-Touch快捷访问
date: 2018-09-14 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - 3d-touch
---

#用法

## 添加快捷项(UIApplicationShortcutItem)
有两种途径, 编辑`Info.plist`或代码添加

### Info.plist

```xml
<key>UIApplicationShortcutItems</key>
<array>
	<dict>
	   <!--图标, 必须-->
		<key>UIApplicationShortcutItemIconType</key>
		<string>UIApplicationShortcutIconTypeCapturePhoto</string>
		<!--标题, 必须-->
		<key>UIApplicationShortcutItemTitle</key>
		<string>Scan</string>
		<!-副标题-->
		<key>UIApplicationShortcutItemSubtitle</key>
		<string>QR Code</string>
		<!--快捷项标识符-->
		<key>UIApplicationShortcutItemType</key>
		<string>$(PRODUCT_BUNDLE_IDENTIFIER).Scan</string>
	</dict>
</array>
```
完整可选项见[文档](https://developer.apple.com/library/content/documentation/General/Reference/InfoPlistKeyReference/Articles/iPhoneOSKeys.html)

### 代码添加
```swift
// Construct the items.
let shortcut3 = UIMutableApplicationShortcutItem(
	type: ShortcutIdentifier.Third.type, 
	localizedTitle: "Play", 
	localizedSubtitle: "Will Play an item", 
	icon: UIApplicationShortcutIcon(type: .play), 
	userInfo: [
        AppDelegate.applicationShortcutUserInfoIconKey: UIApplicationShortcutIconType.play.rawValue
    ]
)
    
let shortcut4 = ... // 同上
    
// Update the application providing the initial 'dynamic' shortcut items.
application.shortcutItems = [shortcut3, shortcut4]
```
## 良好实践

1. 实现一个`(BOOL)handleShortcutItem:(UIApplicationShortcutItem *)shortcutItem`返`BOOL`值的方法, 里面进行业务操作
2. 实现代理方法:

```
- (void)application:(UIApplication *)application performActionForShortcutItem:(UIApplicationShortcutItem *)shortcutItem completionHandler:(void (^)(BOOL))completionHandler {
    completionHandler([self handleShortcutItem:shortcutItem]);
}
```

3. 在`didBecomeActive`方法里判断是否需要 handle 快捷方式

```Objective-C
- (void)applicationDidBecomeActive:(UIApplication *)application {
    if(!self.launchedShortcutItem) return;
    [self handleShortcutItem:self.launchedShortcutItem];
    self.launchedShortcutItem = nil;
}
```

4. 3说明如果你需要提取一个属性`launchedShortcutItem`
5. 如果提取了属性, 那么`didFinishLaunch`也可以顺便改为:

```Objective-C
BOOL shouldPerformAdditionalDelegateHandling = YES;
UIApplicationShortcutItem *shortcutItem = (UIApplicationShortcutItem *)launchOptions[UIApplicationLaunchOptionsShortcutItemKey];
if(shortcutItem) {
    self.launchedShortcutItem = shortcutItem;
    shouldPerformAdditionalDelegateHandling = NO;
}

// 你的其它初始代码
   
return shouldPerformAdditionalDelegateHandling;  // 通常这里返的是 YES;
```

试试吧

#参考资料

1. [官方文档](https://developer.apple.com/library/content/documentation/UserExperience/Conceptual/Adopting3DTouchOniPhone/)
2. [示例代码](https://developer.apple.com/library/content/samplecode/ApplicationShortcuts/Introduction/Intro.html#//apple_ref/doc/uid/TP40016545)
3. [快捷图标](https://developer.apple.com/documentation/uikit/uiapplicationshortcuticontype)
4. [模拟器支持](https://github.com/DeskConnect/SBShortcutMenuSimulator)
5. [iOS Keys](https://developer.apple.com/library/content/documentation/General/Reference/InfoPlistKeyReference/Articles/iPhoneOSKeys.html) 一些键值的说明
