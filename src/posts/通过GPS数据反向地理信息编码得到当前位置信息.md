---
layout: post
title: 通过GPS数据反向地理信息编码得到当前位置信息
slug: 通过GPS数据反向地理信息编码得到当前位置信息
date: 2018-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - gps
  - geo
---

# 检查可用性
这属于基础知识, 不赘述, 总的来说,你的设备的支持要打开, 添加CoreLocation的framework, 引用头文件, 添加委托,然后, 好的实践是在使用前编程检查相关可用性:
```
- (CLLocationManager *)locationManager
{
    if(!_locationManager){
        if([CLLocationManager locationServicesEnabled]){
            _locationManager = [[CLLocationManager alloc] init];
            _locationManager.delegate = self;
            _locationManager.desiredAccuracy = kCLLocationAccuracyHundredMeters;
            CLAuthorizationStatus status = [CLLocationManager authorizationStatus];
            if (status == kCLAuthorizationStatusNotDetermined) {
                NSLog(@" not determined");
                if([_locationManager respondsToSelector:@selector(requestWhenInUseAuthorization)]){
                    [_locationManager requestAlwaysAuthorization];
                }
            }else if (status == kCLAuthorizationStatusDenied) {
                NSLog(@"denied");
            }else if (status == kCLAuthorizationStatusRestricted) {
                NSLog(@"restricted");
            }else if (status == kCLAuthorizationStatusAuthorizedAlways) {
                NSLog(@"always allowed");
            }else if (status == kCLAuthorizationStatusAuthorizedWhenInUse) {
                NSLog(@"when in use allowed");
            }else{
            }
        }else _locationManager = nil;
    }
    return _locationManager;
}
```
注意`kCLAuthorizationStatusNotDetermined`状态, iOS8以后, 需要手动编辑info.plist文件, 添加两个请求用户授权时的文案, 才能正常使用, 这里觉得匪夷所思:
```
<key>NSLocationWhenInUseUsageDescription</key><string>请授权使用地理位置服务</string>
<key>NSLocationAlwaysUsageDescription</key><string>请授权使用地理位置服务</string>
```
以上, 可随便参考网上任何[一篇教程](http://kittenyang.com/cllocationmanager/)

# 请求地理位置并反向编码

这里需要注意的是, 苹果的`CLGeocoder` API并不允许你频繁调用, 一分钟一次为宜, 所以你千万不要`[self.locationManager startUpdatingLocation]`, 然后在`locationManager:didChangeAuthorizationStatus:`
方法里去decode, 因为只是为了获取城市, 精度要求不高, 并且不需要持续更新, 所以我们就不update了, 只request一次, 然后在获取位置失败的时候再手动request一次:
```
+ (void)locationManager:(nonnull CLLocationManager *)manager didFailWithError:(nonnull NSError *)error{
    NSLog(@"fail with error:\n %@", error);
    [self.locationManager requestLocation];
}
```
相关解释参考[这篇文章](http://stackoverflow.com/questions/17867422/kclerrordomain-error-2-after-geocoding-repeatedly-with-clgeocoder)

#语言的问题

因为习惯用英文系统, 就碰到请求回来的信息是英文的原因, 这里苹果是固化起来的, 暂时不支持用参数来指定返回数据的显示语言, 借鉴[这篇文章](http://stackoverflow.com/questions/20388891/cllocationmanager-reversegeocodelocation-language)的思路, 在请求前把当前语言设置保存起来, 临时改成中文, 请求结束后再修改回来:
```
+ (void)locationManager:(nonnull CLLocationManager *)manager didUpdateLocations:(nonnull NSArray *)locations{
    CLLocation *location = [locations lastObject];
    CLGeocoder *geocoder = [CLGeocoder new];
    // 修改语言为中文
    NSArray *currentLanguageArray = [[NSUserDefaults standardUserDefaults] objectForKey:@"AppleLanguages"];
    [[NSUserDefaults standardUserDefaults] setObject: [NSArray arrayWithObjects:@"zh_Hans", nil] forKey:@"AppleLanguages"];
    [geocoder reverseGeocodeLocation:location completionHandler:^(NSArray<CLPlacemark *> * __nullable placemarks, NSError * __nullable error) {
        // 恢复语言
        [[NSUserDefaults standardUserDefaults] setObject:currentLanguageArray forKey:@"AppleLanguages"];
        if(error){
            NSLog(@"reverse error:%@", [error localizedDescription]);
        }else{
            if([placemarks count] > 0){
                CLPlacemark *mark = [placemarks firstObject];
                NSLog(@"%@", mark);
                NSLog(@"城市名:%@", mark.locality);
            }
        }
    }];
}
```
