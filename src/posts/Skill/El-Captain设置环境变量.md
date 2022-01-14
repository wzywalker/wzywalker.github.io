---
layout: post
title: El-Captain设置环境变量
slug: El-Captain设置环境变量
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - Skill
---

这里说的不是设置变量给bash/shell来用, 而是给程序使用, 比如, chromium自36版以后, 就不再内置google api keys, [官方文档](http://www.chromium.org/developers/how-tos/api-keys)说明你打包的时候没有添加key的话, 可以在runtime添加, 比如在系统的环境变量里添加进去.

>Providing Keys at Runtime
If you prefer, you can build a Chromium binary (or use a pre-built Chromium binary) without API keys baked in, and instead provide them at runtime. To do so, set the environment variables GOOGLE_API_KEY, GOOGLE_DEFAULT_CLIENT_ID and GOOGLE_DEFAULT_CLIENT_SECRET to your "API key", "Client ID" and "Client secret" values respectively.

至于key哪来的请自行google, 我们不去申请key的话, 还是拿来主义:

>export GOOGLE_API_KEY="AIzaSyCkfPOPZXDKNn8hhgu3JrA62wIgC93d44k"
export GOOGLE_DEFAULT_CLIENT_ID="811574891467.apps.googleusercontent.com"
export GOOGLE_DEFAULT_CLIENT_SECRET="kdloedMFGdGla2P1zacGjAQh"

关于如何在mac上设置环境变量, 有这么[一篇雄文](http://www.dowdandassociates.com/blog/content/howto-set-an-environment-variable-in-mac-os-x-terminal-only/), 我一般是直接编辑`~/.bash_profile`文件, 这次不生效了, 改来改去都没用, 于是换关键词: `yosemite/el captain下如何设置环境变量`, 立刻就有[答案](http://stackoverflow.com/questions/25385934/setting-environment-variables-via-launchd-conf-no-longer-works-in-os-x-yosemite)了

头两个答案都可以, 第一个是使用了`setenv VARIABLENAME=VALUE`这种语法, 第二个是直接在一个文件里编辑, 然后使之生效, 我直接用了第二种, 因为文本随时可编辑, 可查看

1, Create an environment.plist file in `~/Library/LaunchAgents/` with this content:
```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>my.startup</string>
  <key>ProgramArguments</key>
  <array>
    <string>sh</string>
    <string>-c</string>
    <string>
        launchctl setenv GOOGLE_API_KEY AIzaSyCkfPOPZXDKNn8hhgu3JrA62wIgC93d44k
        launchctl setenv GOOGLE_DEFAULT_CLIENT_ID 811574891467.apps.googleusercontent.com
        launchctl setenv GOOGLE_DEFAULT_CLIENT_SECRET kdloedMFGdGla2P1zacGjAQh
    </string>
  </array>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
```
2, You can add many launchctl commands inside the `<string></string>` block.可见, 我们只需要在`string`标签里写需要的内容就行了, 本例是一系列google api keys.

3, The plist will activate after system reboot. You can also use `launchctl load ~/Library/LaunchAgents/environment.plist` to launch it immediately.
