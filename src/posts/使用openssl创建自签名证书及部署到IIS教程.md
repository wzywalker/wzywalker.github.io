---
layout: post
title: 使用openssl创建自签名证书及部署到IIS教程
slug: 使用openssl创建自签名证书及部署到IIS教程
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

#创建自签名证书
首先，创建一个私钥文件：
```
openssl genrsa -out myselfsigned.key 2048
```
然后利用私钥创建自签名证书：
```
openssl req -new -x509 -key myselfsigned.key -out myselfsigned.cer -days 36500
```
执行上面的两个操作之后会提示输入以下几个内容(为了显示正常尽量使用英文)：
>1. Country Name (2 letter code) [AU]:CN //国家简称
1. State or Province Name (full name) [Some-State]:GuangDong //州或省的名字
1. Locality Name (eg, city) []:ShenZhen //区或市县的名称
1. Organization Name (eg, company) [Internet Widgits Pty Ltd]:Comapny //公司或组织名
1. Organizational Unit Name (eg, section) []:Mobile //单位或者是部门名称
1. Common Name (e.g. server FQDN or YOUR name) []:xxxxxx //域名或服务器名或IP
1. Email Address []:xxxxx@gmail.com //Email地址

注, 上述可直接在命令中用`-subj`跟在语句后面, 如:
```
openssl req -new -x509 -key myselfsigned.key -out myselfsigned.cer -days 36500 -subj /CN=域名或服务器名或IP
```
至此, 生成的myselfsigned.cer分别应用到服务器端以及客户端(通过邮件, 链接等方式下发), 即可使用, 配置IIS见下文 

#创建自己的证书颁发机构(CA)
即使是测试目的, 也会出现有多个站点需要自定义证书的情况, 不可能要求用户每个站点装一个 我们何不把自己添加成一个证书颁发机构(CA), 然后把这个证书装给客户端, 那么由这个CA颁发的证书都会被自动信任. 

首先, 用同样的语法创建一个证书, 我们把名字取明确一些, 就叫`myCA`吧(跟第一步生成普通证书是一样一样的, 只是这次我们把它理解成一个证书颁发机构)
```
openssl genrsa -out myCA.key 2048
openssl req -new -x509 -key myCA.key -out myCA.cer -days 36500
```
然后, 基于这个证书生成一个证书请求(`CSR`), (同样, 先生成一个key, 要用key来请求)
```
openssl genrsa -out server.key 2048
openssl req -new -out server.req -key server.key -subj /CN=域名
```
>注:
1. 一旦域名配置了, 用不同于这个域名的主机名来请求, 就会校验失败
2. 这里用到了上面说的-subj参数

最后, 通过服务器证书(我们理解的CA), 对这个签发请求进行签发
```
openssl x509 -req -in server.req -out server.cer -CAkey myCA.key -CA myCA.cer -days 36500 -CAcreateserial -CAserial serial
 ```

#配置IIS
我们的使用场景是IIS伺服了一个静态文件服务器(没错, 是用来放iOS企业部署的的plist和ipa文件的), 做到如下几步

##转化证书格式
IIS导入证书需要转化为pkcs12格式(X509格式?), 中间会询问一次密码, 牢记, 或者与导出的文件一起保存
```
openssl pkcs12 -export -clcerts -in server.cer -inkey server.key -out iis.pfx
```
现在总结一下, 目前为止, 除去`key`和`car`, 生成了`myCA.cer`, `server.cer` 和`iis.pfx`三个文件 

##将myCA.cer添加为”受信任的根证书颁发机构”
打开IE > 工具 > Internet选项 > 内容 > 证书 > 受信任的根证书颁发机构 > 导入 > 选择iis.pfx > 输入密码 > 导入

##添加服务器证书
这需要两个步骤

首先, 在IIS管理器(`inetmgr`)的根目录上(就是机器名), 选择”服务器证书”, 导入我们刚才用`server.cer`生成的`iis.pfx`, 即给IIS添加了一个证书(如果有多个, 重复以上步骤)

然后, 找到网站节点, 右键, “编辑绑定”, 添加一个供https访问的端口(默认是443), 此时会要求你选择一个证书, 把刚才通过管理器添加的证书名选出来, 即可.

最后, 把`server.cer`通用你们企业自己的方式颁发给需要使用的客户端(邮件, 链接等, 均可), 如果是iPhone, 点击了`server.cer`文件后, 会导航到设置里面安装, 安装并信任后, 在设置 > 通用 > Profiles里面可以看到你信任的证书使用openssl创建自签名证书及部署到IIS教程
