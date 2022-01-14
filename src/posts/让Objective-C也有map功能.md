---
layout: post
title: 让Objective-C也有map功能
slug: 让Objective-C也有map功能
date: 2019-07-30 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - map
  - keypath
  - kvc
---

map一个数组是大部分高级语言都有的, OC 没有, 有几个方案让它实现, 我优选出三个: 

## 原生实现

其实就是`valueForKeyPath`的活用:

```objective-c
NSArray *names = [allEmployees valueForKeyPath: @"[collect].{daysOff<10}.name"];
NSArray *albumCovers = [records valueForKeyPath:@"[collect].{artist like 'Bon Iver'}.<NSUnarchiveFromDataTransformerName>.albumCoverImageData"];
```

## category

这个大家肯定早想到过了, 你没有, 我给你扩展出来一个:

定义:

```objective-c
@interface NSArray (Map)

- (NSArray *)mapObjectsUsingBlock:(id (^)(id obj, NSUInteger idx))block;

@end

@implementation NSArray (Map)

- (NSArray *)mapObjectsUsingBlock:(id (^)(id obj, NSUInteger idx))block {
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:[self count]];
    [self enumerateObjectsUsingBlock:^(id obj, NSUInteger idx, BOOL *stop) {
        [result addObject:block(obj, idx)];
    }];
    return result;o
}

@end
```

使用:

```objective-c
NSArray *people = @[
                     @{ @"name": @"Bob", @"city": @"Boston" },
                     @{ @"name": @"Rob", @"city": @"Cambridge" },
                     @{ @"name": @"Robert", @"city": @"Somerville" }
                  ];
// per the original question
NSArray *names = [people mapObjectsUsingBlock:^(id obj, NSUInteger idx) {
    return obj[@"name"];
}];
// (Bob, Rob, Robert)

// you can do just about anything in a block
NSArray *fancyNames = [people mapObjectsUsingBlock:^(id obj, NSUInteger idx) {
    return [NSString stringWithFormat:@"%@ of %@", obj[@"name"], obj[@"city"]];
}];
// (Bob of Boston, Rob of Cambridge, Robert of Somerville)
```

## 三方库

是的, 一般简单功能能自己实现就自己实现, xcode 项目还是不能像 `nodejs`项目一样, 哪怕有的包里也只有一句话, 我也要从用第三方的...ddddd

js 的世界里, `underscore`用来处理数组可算神器, 自然, 我也挑中了同样名字的 OC 库: [Underscore](http://underscorem.org/)

```objective-c
NSArray *tweets = Underscore.array(results)
    // Let's make sure that we only operate on NSDictionaries, you never
    // know with these APIs ;-)
    .filter(Underscore.isDictionary)
    // Remove all tweets that are in English
    .reject(^BOOL (NSDictionary *tweet) {
        return [tweet[@"iso_language_code"] isEqualToString:@"en"];
    })
    // Create a simple string representation for every tweet
    .map(^NSString *(NSDictionary *tweet) {
        NSString *name = tweet[@"from_user_name"];
        NSString *text = tweet[@"text"];

        return [NSString stringWithFormat:@"%@: %@", name, text];
    })
    .unwrap;
```

> 当然, 所有方案来源于[StackOverflow](http://stackoverflow.com/questions/6127638/nsarray-equivalent-of-map) 上的答案, 一些其它方案, 其它库(如[BlocksKit](http://cocoadocs.org/docsets/BlocksKit/2.2.3/Categories/NSArray+BlocksKit.html#//api/name/bk_map:)), 都可以试试, 也挺简洁的
