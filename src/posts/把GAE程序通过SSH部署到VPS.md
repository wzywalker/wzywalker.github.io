---
layout: post
title: 把GAE程序通过SSH部署到VPS
slug: 把GAE程序通过SSH部署到VPS
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - Skill
---

大部分在[文档](https://cloud.google.com/appengine/downloads)上都写了, 写这篇文章的目的是发现现在`appcfg.py update xxxx`的时候会打开浏览器访问google请求授权(后台内建了一个本地server, 端口是`8090`, 授权成功后会带授权码请求本地的8090端口, 所以我们在 ssh 环境中是没有浏览器的, 该怎么解决呢?

我玩 linux 不多, 就以我的知识量这么解决:

1. 碰到需要访问网站的时候, 程序已经给出了提示, 要你退出, 你当然不退出, 而是把网址复制出来, 在本地打开, 授权成功后, 本地浏览器会请求`127.0.0.1:8090`, 当然, 什么都不会发生, 但从地址栏里把地址复制到剪贴板.
2. 回到SSH, 把当前任务放到后台(`ctrl+z`)
3. 用`curl`访问剪贴板里的地址
4. 继续`ctrl+z`把`curl`请求放到后台
5. `jobs`命令查一下, 如果后台没有别的任务的话, `appcfg`任务的id 应该是1, `curl`任务id 应该是2(现在以我的1和2为准)
6. 把`appcfg`提到前台: `fg %1`
7. 你会看到程序顺利进行下去了
8. 继续, `fg %2`把 curl 任务提到前台, 你会看到提示, 什么授权成功之类的

演示:

    root@walker:~/KindleEar# appcfg.py update app.yaml module-worker.yaml
    07:52 AM Host: appengine.google.com
    07:52 AM Application: kindleearwalker; version: 1
    07:52 AM Starting update of app: kindleearwalker, version: 1
    07:52 AM Getting current resource limits.
    Your browser has been opened to visit:
    
        https://accounts.google.com/o/oauth2/auth?scope=演示数据
        # step1: 请复制此网址, 并忽略要你退出换电脑的提示
    
    If your browser is on a different machine then exit and re-run this
    application with the command-line parameter
    
      --noauth_local_webserver
    
    # step2: 现在开始把任务放到后台
    ^Z
    [1]+  Stopped                 appcfg.py update app.yaml module-worker.yaml
    # step3: 把从本机浏览器复制的回调 url 访问一下
    root@walker:~/KindleEar# curl http://localhost:8090/?code=4/CYdQFQLiLBFwa7ajsU1acb1Xx9Kpal6SxMuPIS-dRYo#
    # step4: 把访问任务放到后台
    ^Z
    [2]+  Stopped                 curl http://localhost:8090/?code=4/CYdQFQLiLBFwa7ajsU1acb1Xx9Kpal6SxMuPIS-dRYo#
    # step 5: 查看一下任务和 ID
    root@walker:~/KindleEar# jobs
    [1]-  Stopped                 appcfg.py update app.yaml module-worker.yaml
    [2]+  Stopped                 curl http://localhost:8090/?code=4/xxxxx#
    # step 6: 把appcgf的任务提到前台
    root@walker:~/KindleEar# fg %1
    appcfg.py update app.yaml module-worker.yaml
    Authentication successful.
    07:54 AM Scanning files on local disk.
    07:54 AM Cloning 15 static files.
    07:54 AM Cloning 387 application files.
    07:54 AM Uploading 3 files and blobs.
    07:54 AM Uploaded 3 files and blobs.
    .........省略
    07:54 AM Compilation completed.
    07:54 AM Starting deployment.
    07:54 AM Checking if deployment succeeded.
    07:54 AM Deployment successful.
    07:54 AM Checking if updated app version is serving.
    07:54 AM Completed update of app: kindleearwalker, module: worker, version: 1
    # step 7: see? 成功了, 看看剩下的任务吧
    root@walker:~/KindleEar# jobs
    [2]+  Stopped                 curl http://localhost:8090/?code=4/xxxxxxx#
    # step 8: 提到前台来结束吧
    root@walker:~/KindleEar# fg %2
    curl http://localhost:8090/?code=4/CYdQFQLiLBFwa7ajsU1acb1Xx9Kpal6SxMuPIS-dRYo#
    <html><head><title>Authentication Status</title></head><body><p>The authentication flow has completed.</p></body></html>root@walker:~/KindleEar#
