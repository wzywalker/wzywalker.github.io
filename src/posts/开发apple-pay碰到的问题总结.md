---
layout: post
title: 开发apple-pay碰到的问题总结
slug: 开发apple-pay碰到的问题总结
date: 2019-03-04 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - apple pay
  - debug
---

本来想简单总结一下Apple Pay 开发过程中的几个问题, 结果被[这篇文章](http://www.lilongcnc.cc/2016/02/28/9-applepay支付界面调用和获取payment参数银联扣款流程/)全碰上了, 干脆全文转载, 作者对相关资源整理得比较详细, 比较有参考价值, 建议阅读, 我做个概述.

总的来说, 我们做过 APNs 推送的话, 申请 商户ID 并关联到 apple id, 申请证书, 生成provisioning profile等步骤都差不多

然后我真机调试有两个地方没通过, 下文也总结了, 我拎出来单独说一下:

1, Payment request is invalid: check your entitlements. Connection to remote alert view service failed
>原因:
粗心, 把merchant id写错了.
之所以要把粗心的事也列出来, 是因为, 我出问题是粗心, 但是因为集成苹果支付的过程中, 是需要在配置界面的`Capabilities`里面用下拉列表选择一个`merchant id`, 以及代码里还要写一次的, 如果你有多个`merchant id`, 或者开发过程中切换过, 下拉列表值和代码里手写的值要记得同步, 没有同步, 一样会得上上面的错误

2, 进不到didAuthorizePayment方法.
>原因:
`payrequest.merchantCapabilities  = PKMerchantCapability3DS|PKMerchantCapabilityEMV`. 
看到了吧, 后面的 EMV 是必须要加的
大部分碰到同样问题的同学估计都是看 WWDC 的视频, 里面的小哥说3DS 是必须的, 显然在咱们大天朝, EMV 也是必须的.
