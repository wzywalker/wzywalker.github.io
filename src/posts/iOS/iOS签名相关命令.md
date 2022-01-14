---
layout: post
title: iOS签名相关命令
slug: iOS签名相关命令
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

用openssl查看CSR文件内容
```
openssl asn1parse -i -in CertificateSigningRequest.certSigningRequest
```
查看下发的证书的内容:
```
openssl x509 -inform der -in ios_development.cer -noout -text
```
可以使用如下命令查看一个mobileprovision：
```
security cms -D -i embedded.mobileprovision
```
ipa文件是一个zip包，可以使用如下命令解压：
```
/usr/bin/unzip -q xxx.ipa -d <destination>
```
用下面命令，列出系统中可用于签名的有效证书：
```
/usr/bin/security find-identity -v -p codesigning
```
使用如下命令对xxx.app目录签名，codesign程序会自动将其中的文件都签名，（Frameworks不会自动签）：
```
/user/bin/codesign -fs "iPhone Developer: Your Cert Name (VDT388662Q)" --no-strict Payload/xxx.app
```
最后用下面命令校验签名是否合法：
```
/usr/bin/codesign -v xxx.app
```
使用zip命令重新打包成ipa包
```
/usr/bin/zip -qry destination source
```
[来源]([https://zhuanlan.zhihu.com/p/53006952](https://zhuanlan.zhihu.com/p/53006952)
)
