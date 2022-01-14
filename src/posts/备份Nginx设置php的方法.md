---
layout: post
title: 备份Nginx设置php的方法
slug: 备份Nginx设置php的方法
date: 2019-07-31 00:00
status: publish
author: walker
categories: 
  - Skill
tags:
  - nginx
  - php
---

版本nginx/1.12.2
```
  location ~ \.php$ {
    fastcgi_pass   unix:/run/php/php7.0-fpm.sock;
    fastcgi_param  SCRIPT_FILENAME $document_root/$fastcgi_script_name;
    include        fastcgi_params;
   }
```
我机器上的fastcgi_param值不对, 改成上图成功, 备份下
