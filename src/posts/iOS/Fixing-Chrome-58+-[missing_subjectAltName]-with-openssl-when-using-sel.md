---
layout: post
title: Fixing-Chrome-58+-[missing_subjectAltName]-with-openssl-when-using-sel
slug: Fixing-Chrome-58+-[missing_subjectAltName]-with-openssl-when-using-sel
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

[原文链接](https://alexanderzeitler.com/articles/Fixing-Chrome-missing_subjectAltName-selfsigned-cert-openssl/)

# 说在前面

>1. `createselfsignedcertificate.sh`文件里的sudo删掉了
2. `server.csr.cnf`里`dn`里面的内容请改成自己的
3. `v3.ext`里面的DNS.1也更改为自己的server
4. 本来我只想绑一个固定的 IP, 基本通过, 但是在mac的chrome58下, 仍然过不了, 最终还是通过域名解决

上一个在 chrome58下终于变绿的图片
![](../assets/1859625-c95f8edc4f1abe55.png)

# 原文转载

Since version 58, Chrome requires SSL certificates to use SAN (Subject Alternative Name) instead of the popular Common Name (CN), thus [CN support has been removed](https://groups.google.com/a/chromium.org/forum/#!msg/security-dev/IGT2fLJrAeo/csf_1Rh1AwAJ).If you're using self signed certificates (but not only!) having only CN defined, you get an error like this when calling a website using the self signed certificate:
![](../assets/1859625-1e986c014d7a82fe.png)
Here's how to create a self signed certificate with SAN using openssl

First, lets create a root CA cert using `createRootCA.sh`:

```bash
#!/usr/bin/env bash
mkdir ~/ssl/openssl genrsa -des3 -out ~/ssl/rootCA.key 2048
openssl req -x509 -new -nodes -key ~/ssl/rootCA.key -sha256 -days 1024 -out ~/ssl/rootCA.pem
```

Next, create a file `createselfsignedcertificate.sh`:

```bash
#!/usr/bin/env bash
openssl req -new -sha256 -nodes -out server.csr -newkey rsa:2048 -keyout server.key -config <( cat server.csr.cnf )
openssl x509 -req -in server.csr -CA ~/ssl/rootCA.pem -CAkey ~/ssl/rootCA.key -CAcreateserial -out server.crt -days 500 -sha256 -extfile v3.ext
```

Then, create the openssl configuration file `server.csr.cnf`
 referenced in the openssl command above:

```
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn

[dn]
C=US
ST=New York
L=Rochester
O=End Point
OU=Testing Domain
emailAddress=you@example.com
CN = localhost
```

Now we need to create the `v3.ext` file in order to create a X509 v3 certificate instead of a v1 which is the default when not specifying a extension file:

```
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
```

In order to create your cert, first run `createRootCA.sh` which we created first. Next, run `createselfsignedcertificate.sh` to create the self signed cert using localhost as the SAN and CN.
After adding `the rootCA.pem` to the list of your trusted root CAs, you can use the `server.key` and `server.crt` in your web server and browse https://localhost using Chrome 58 or later:
![](../assets/1859625-ceb520b0fa9c7b1f.jpg)
You can also verify your certificate to contain the SAN by calling

```
openssl x509 -text -in server.crt -noout
```

Watch for this line `Version: 3 (0x2)` as well as `X509v3 Subject Alternative Name:` (and below).
Happy self signing!
