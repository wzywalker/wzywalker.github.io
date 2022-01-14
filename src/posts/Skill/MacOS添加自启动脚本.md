---
layout: post
title: MacOS添加自启动脚本
slug: MacOS添加自启动脚本
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - Skill
---

MacOS下添加自启动脚本有很多方法, 在一篇[知乎文章](https://www.zhihu.com/question/22794908/answer/89421030)中了解到Launchd替代了过去的init, rc, init.d, rc.d, SystemStarter, inted/xinetd, watchdogd等, 建议用Launchd.
当然还有别的Automator, Apple Script等方式(底层未研究), 感兴趣的自己搜索, 我选择了直接Launchd, 结合so上的[这篇文章](https://stackoverflow.com/a/13372744/1051235):

1. 编写自己的脚本, 添加可执行权限`chmod a+x myscript.sh`
2. 编写Launchd配置文件(`.plist`文件)
3. 结合上述两篇文章, 确定在系统启动还是用户启动时运行脚本, 我选择的是用户目录(`~/Library/LaunchAgents/`)
4. load这个配置: `launchctl load -w ~/Library/LaunchAgents/com.service.name.plist`
5. 登入登出测试, 或: `launchctl start com.service.name`

注:
1. 可执行脚本里的路径有空格需要转义
2. 但plist文件里`<string>`标签里的目录如果有空格, 不需要转义
3. `load`带`-w`参数参见[这篇文章](https://apple.stackexchange.com/a/308421)
4. 如果出错, 运行`Console`应用查看日志, 或参考[这篇文章](https://stackoverflow.com/a/48017581), 定向日志输出文件
即在`.plist`文件里添加:
```xml
<key>StandardOutPath</key>
<string>/var/log/mylog.log</string>
```

附: `.plist`文件示例
```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
	<dict>
		<key>Label</key>
		<string>com.service.name</string>
		<key>ProgramArguments</key>
		<array>
			<string>/path/to/my/script.sh</string>
		</array>
		<key>RunAtLoad</key>
		<true/>
	</dict>
</plist>
```

如果执行的脚本就一句话, 你可能希望直接在`.plist`文件里运行, 而不是额外再多生成一个脚本吧? ([source](https://superuser.com/a/285273))
```
<key>ProgramArguments</key>
<array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>ls -1 | grep *.txt | echo &gt; allTextFiles</string>
</array>
```

继续, 如果还想[以root来执行脚本](https://superuser.com/questions/36087/how-do-i-run-a-launchd-command-as-root), 综合起来, 我的实现如下:
```bash
cp com.run.udp2raw.plist /Library/LaunchDaemons
cd /Library/LaunchDaemons
sudo launchctl load -w com.run.udp2raw.plist
sudo launchctl start com.run.udp2raw
```
其中`udp2raw`对应的命令是需要`root`权限的, 实测通过. 我选择的是`/Library/LaunchDaemons/`
>注: 唯一要注意的地方, 就是最后两行, `load`和`start`命令都需要加`sudo`. 没有加的时候没有报错, 但是没有运行成功.

附: folders and usage
```
|------------------|-----------------------------------|---------------------------------------------------|
| User Agents      | ~/Library/LaunchAgents            | Currently logged in user
|------------------|-----------------------------------|---------------------------------------------------|
| Global Agents    | /Library/LaunchAgents             | Currently logged in user
|------------------|-----------------------------------|---------------------------------------------------|
| Global Daemons   | /Library/LaunchDaemons            | root or the user specified with the key UserName
|------------------|-----------------------------------|---------------------------------------------------|
| System Agents    | /System/Library/LaunchAgents      | Currently logged in user
|------------------|-----------------------------------|---------------------------------------------------|
| System Daemons   | /System/Library/LaunchDaemons     | root or the user specified with the key UserName
|------------------|-----------------------------------|---------------------------------------------------|
```
