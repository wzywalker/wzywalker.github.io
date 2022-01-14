---
layout: post
title: 将共享文件夹用作Time-Machine并加密
slug: 将共享文件夹用作Time-Machine并加密
date: 2019-11-21 00:00
status: publish
author: walker
categories: 
  - iOS
tags:
  - time machine
  - nas
---

怎么在局域网创建一个共享文件夹不在此文讨论范围内, 比如windows文件夹简单右键共享一下, 就能走完本教程. 以下是在macOS上设置Time Machine的操作.

#Step 1: 各种命名
没什么用的第一步, 如果你有多台电脑, 那就最好用名字和MAC地址来作备份的名字, 送佛送到西:

	MAC_ADDRESS=`ifconfig en0 | grep ether | awk '{print $2}' | sed 's/://g'`
	SHARE_NAME=`scutil --get ComputerName`
	IMG_NAME=${SHARE_NAME}_${MAC_ADDRESS}.sparsebundle
	echo $IMG_NAME
	
#Step 2: 创建并加密一个镜像
复制粘贴前记得更改一下'MAXSIZE', 设为自己想要的大小. 为了怕人不看文字直接复制, 我设定了一个合理的350G

	MAXSIZE=350g
	hdiutil create -size $MAXSIZE -type SPARSEBUNDLE -nospotlight -volname "Backup of $SHARE_NAME" -fs "Case-sensitive Journaled HFS+" -verbose unencrypted_$IMG_NAME
	hdiutil convert -format UDSB -o "$IMG_NAME" -encryption AES-128 "unencrypted_$IMG_NAME"
	rm -Rf "unencrypted_$IMG_NAME"
注意两点:
1. 该脚本先创建了一个未加密的image(其实是一个文件夹), 随后加密, 过程中会询问密码, 最后删除未加密的image
2. 文件会创建在用户主目录, 如果空间不够, 可以读一下`hdiutil`的文档, 自行设定到远程共享文件夹去. 如果按本脚本, 那么请自行移动到共享目录

#Step 3: 设置Time Machine
双击共享文件夹里的镜像, 输入上一步设置的密码, 此时会mount到本地, 菜单栏上的Time Machine的选择备份文件夹功能里应该能看到这个盘, 但是你不能用它, 我们用命令来关联:

	defaults write com.apple.systempreferences TMShowUnsupportedNetworkVolumes 1
	sudo tmutil setdestination "/Volumes/Backup of $SHARE_NAME"
	
此时再打开时光机器, 就可以看到已经自动关联上了(你无需去选择备份硬盘).
有一个小问题, 就是即使我这么操作下来, 即使mount的时候需要输入密码, 备份的时候还是提示往一个没有加密的盘里备份. 也就是说, 我们以为encrypt了, 只是对image而言, 备份还是不加密的. 可见我们还是没有找到像一些NAS系统里那样能被自动发现, 正常加密的方案

>参考:
[source](https://chester.me/archives/2013/04/a-step-by-step-guide-to-configure-encrypted-time-machine-backups-over-a-non-time-capsule-network-share.html/)  
[create an sparse image](http://www.levelofindirection.com/journal/2009/10/10/using-a-networked-drive-for-time-machine-backups-on-a-mac.html)  
[encrypt it](http://www.cognizo.com/2012/04/encrypted-network-backups-with-os-x-time-machine/)  
[convince Time Machine to use it](http://basilsalad.com/how-to/create-time-machine-backup-network-drive-lion/)

>备注: 请活学活用, 比如我就没用那些名字变量, 直接写死了镜像路径和文件名
