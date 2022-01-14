---
layout: post
title: 树莓派利用Privoxy,Shadowsocks,Kcptun做http代理排坑记录
slug: 树莓派利用Privoxy,Shadowsocks,Kcptun做http代理排坑记录
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - Skill
---

我用树莓派做`翻墙网关`, `透明网关`, 通通绕到坑里出不来了, 方案很多, 思路是树莓派翻, 智能dns, 智能国内外分流, 外加让树莓派成为局域网的`网关`以便让局域网用户`无感翻墙`.  
黑科技没玩转, 于是我整理思绪, 做一个最简单的局域网http代理服务器吧. 任何网络设备/应用都能找到设http代理的地方, 这样最后只要让树莓派能应用`gfwlist`的规则.   

我们来看一下最终的拓扑:
![](../assets/1859625-53732f608938788b.png)

几点说明:

1. 如果只是Shadowsocks翻墙, 那么这一步做完就收工了. 最简步骤
2. 如果需要转成http代理, 你就再多包一层代理工具, `Prixovy`, `Polipo`等, 用关键词搜, 比, 自己定. `Polipo`配置要简单很多
3. 对, 不考虑做成网关就是这么简单, 但是我还想用`kcptun`给提提速, 所以多插一脚
4. 不需要更多了, 再多就又有新坑了.

开始排坑

# shadowsocks-libev

我使用的是一键安装脚本, 它会给你默认启动成`ss-server`, 显然, 你需要的是`ss-local`, 如果从名字可以看出它们的差别, 当然还有`ss-redir`, 你要搭建`透明`代理的话用着得, 还有`ss-tunnel`, 转发DNS.
>`lsof -i:8530`看看跑8530端口的是哪个程序

我讲解一下怎么把`ss-server`改成`ss-local`. 

1. 它是以`ss-server -c <configFILE> -f <pidFILE>`启动的, 写在了`/etc/init.d/shadowsocks` 里面, 你把所有`ss-server`改为`ss-local`, 然后顺便把`-f`参数删掉
2. 改完后, 保存: `update-rc.d -f shadowsocks defaults`
3. 重启: `/etc/init.d/shadowsocks restart`

如果需要连kcptun, 那么这里是第二个需要注意的地方, 即你的服务器地址本应为`shadowsocks`服务器地址及端口, 这里要改成`127.0.0.1:<kcptun端口>`

shadowsocks就上述两个需要注意的地方.
如果你使用了python版, go版的, 或是手工安装的libev版, 那么`改ss-server`这个坑你就碰不到了

# kcptun

下载`Linux`版的kcptun包, 选择名字里含有`client`和`armv7`的文件即可, 这里无其它坑, 选对文件是关键, 配置文件参数与服务端一致即可.

>网上有些文章说一定要保持server文件和client文件是同一天的, 我实测没必要. 我的server端都部署了一年了, 今天用的最新的client, 没什么问题


# Privoxy

这里坑比较多

1. 按网上普通的教程, 设置自己的监听端口(默认8118)和shadowsocks的(127.0.0.1:1080), 没问题
2. 做完上一步, 我用电脑设置树莓派的8118端口为代理服务器, 居然所有流量**已经**(划重点)被转发了, 而理论上这时是需要你自己添加规则(`.action文件`)的,否则就是直连.
3. 找了很多资料, 没人碰到跟我一样的情况, 所以我是绕了一圈, 做了一个`不转发所有流量`的规则, 然后再在后面跟上我的gfw的规则, 才会选择性地转发. 奇怪
4. 最后就是网上找一个能转`gfwlist`规则的方案应用到`action`里就好, 比如[这个](https://github.com/snachx/gfwlist2privoxy)

分享一个action文件编写的良好实践, 应用别名:

```
{{alias}}
direct   = +forward-override{forward .}
socks    = +forward-override{forward-socks5 localhost:8080 .}
httproxy = +forward-override{forward localhost:8000 .}

{direct}
.google.com
.googleusercontent.com
.mozilla.com
【我就是在这里设了一个*.*, direct了所有的流量】

{socks}
.youtube.com
.ytimg.com

{httproxy}
.twitter.com
.blogspot.com
feedproxy.google.com
```

#总结

1. 树莓派对外暴露`Privoxy`的`8118`端口, 转发至`shadowsocks`的`1080`端口
2. `shadowsocks`转发至`kcptun`的`1087`端口
3. `kcptun client`与`kcptun server`的`29900`通讯
4. `kcptun server`与`shadowsocks server`的`8530`通讯
等于把拓扑图口述了一遍~~~我也是想清了这个才最终思路清晰地做完所有事的

最后, 不管是`privoxy`还是`shadowsocks`还是`kcptun`, 都是需要加入自启动的, 你可以选择在`/etc/rc.local`里面依次写入启动脚本, 也可以在`/etc/init.d/`里面添加对应的脚本文件

本文不是教程, 是排坑指南. 相应的安装, 配置, 加启动, 可以搜既有教程, 如果碰到跟我一样的问题, 希望帮助到了你.

#题外话
我的目的其实是给我的`Apple TV`第3代翻墙, 结果发现它居然不能直接设置http代理, 得用`Apple Configurator 2`来设置, 并推到设备上. 这里有两个选择

1. 设置全局代理(需要设备Supervised)
2. 设置某个WiFi热点的代理

我没去研究什么是`Supervised`了, 而且也希望代理好切换, 于是选择了第二种方案, 即换了wifi后就没代理了(跟在`iPhone`上设置一样)

![](../assets/1859625-f092d16231112699.jpg)
