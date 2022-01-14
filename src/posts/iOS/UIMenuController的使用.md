---
layout: post
title: UIMenuController的使用
slug: UIMenuController的使用
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

#1, 基本使用
以对一个UILabel长按弹出菜单为例
##子类化UILabel
因为需要覆盖这几个方法:
`- (BOOL)canBecomeFirstResponder`; 返回YES 
同时需要在每次UI元素出现的时候去`becomeFirstResponder`一次,才能显示出菜单. 在我的实测中, 我在`ViewDidLoad`里面这么做了, 当UI导航到别的页面(导航控件, 或modal页面), 然后回来, 菜单又失效了, 所以我写到`ViewWillAppear`里面去了, 通过

`- (BOOL)canPerformAction:(SEL)action withSender:(nullable id)sender`;
这个方法会在每一个menuItem生成的时候调用一次, 因此在方法体里就要根据action来判断是否需要显示在菜单里, 如果不需要, 则返回`NO`. 也就是说, 如果你什么都不做, 直接返一个`YES`, 那么所有的默认菜单项都会显示出来, 此处我们只要一个`Copy`选项吧:
```
- (BOOL)canPerformAction:(SEL)action withSender:(id)sender {
    return (action == @selector(copy:));
}
```
##添加触发方式
如果是以长按为触发, 则添加长按手势, 代码片段如下:
```
// 在awakeFromNib里面添加即可
UILongPressGestureRecognizer *menuGesture = [[UILongPressGestureRecognizer alloc] initWithTarget:self action:@selector(menu:)];
    menuGesture.minimumPressDuration = 0.2;
    [self addGestureRecognizer:menuGesture];

- (void)menu:(UILongPressGestureRecognizer *)sender {
    if (sender.state == UIGestureRecognizerStateBegan) {
        UIMenuController *menu = [UIMenuController sharedMenuController];
        [menu setTargetRect:self.frame inView:self.superView]; // 把谁的位置告诉控制器, 菜单就会以其为基准在合适的位置出现
        [menu setMenuVisible:YES animated:YES];
    }
}
```
##编写菜单行为
上面我们只要了copy, 那么就覆盖默认的copy方法:
```
- (void)copy:(id)sender {
    UIPasteboard *paste = [UIPasteboard generalPasteboard];
    paste.string = self.text;
}
```
#2, 添加自定义菜单项
自定义菜单只需要在菜单控制器添加几个item即可, 结合上例, 我的那个label是显示电话号码的, 那么就让它多显示一个”打电话”和一个”发短信”菜单吧, 唯一需要注意的是, 在设置自定义菜单项时, 设置的items只影响自定义部分, 标准菜单项仍然是由`canPerformAction`决定的:
```
UIMenuItem *itemCall = [[UIMenuItem alloc] initWithTitle:@"Call" action:@selector(call:)];
UIMenuItem *itemMessage = [[UIMenuItem alloc] initWithTitle:@"Message" action:@selector(message:)];
[[UIMenuController sharedMenuController] setMenuItems: @[itemCall, itemMessage]];
[[UIMenuController sharedMenuController] update];
```
>注, 添加了两个菜单后, canPerformAction需要相应变化, 自己想想应该怎么改. 也可以在下一节看代码. 当然也要自行写完里面的call和message方法, 参照copy的写法即可

#3, UITableViewCell长按显示菜单
##标准菜单项
UITableView里面长项条目显示标准菜单, 只需要实现下述代理方法即可: 
```
- (BOOL)tableView:(UITableView *)tableView shouldShowMenuForRowAtIndexPath:(NSIndexPath *)indexPath {
    return YES;
}

- (BOOL)tableView:(UITableView *)tableView canPerformAction:(SEL)action forRowAtIndexPath:(NSIndexPath *)indexPath withSender:(id)sender {
    return (action == @selector(copy:)); // 只显示Copy
}

- (void)tableView:(UITableView *)tableView performAction:(SEL)action forRowAtIndexPath:(NSIndexPath *)indexPath withSender:(id)sender {
    if (action == @select(copy:)) {
        UIPasteboard *paste = [UIPasteboard generalPasteboard];
        paste.string = cell.detailLabel.text; // 自行写业务逻辑
    }
}
```
#4, TableViewCell添加自定义菜单项

同样也得子类化一个TableViewCell,目的也是为了覆盖同样的几个方法:
```
- (BOOL)canPerformAction:(SEL)action withSender:(id)sender {
    return (action == @selector(copy:) || action == @selector(call:) || action == @selector(message:)); // 此处我们把三个行为都写全了, 回答上一节的问题
}

- (BOOL)canBecomeFirstResponder {
    return YES;
}
```
但因为tableView已经实现了菜单, 所以不需要显式为每个cell去`becomeFirtResponder`了.

添加菜单项的方法同上, 写菜单行为的方法同`copy:`, 都是一样的.

>注: 你们或许已经发现了, 添加自定义菜单项的时候, 仍然需要`canPerformAction`, 在这里, 与tableView代理里面的同名方法有什么关系? 是的, 两个都要写, tableView里面的只会影响标准菜单, 文档说只支持这两个`UIResponderStandardEditActions` (copy/paste)

>注: 然而, `- (void)tableView:(UITableView *)tableView performAction:(SEL)action forRowAtIndexPath:(NSIndexPath *)indexPath withSender:(id)sender`这个方法却有点别扭, 一来不需要去实现了, 二来又不能注释掉(你们自己试一下), 等于一定要留一个空的方法体在那里…
