---
layout: post
title: 重装mac系统后ssh异常
slug: 重装mac系统后ssh异常
date: 2019-08-03 00:00
status: publish
author: walker
categories: 
  - Skill
tags:
  - macos
  - ssh
---

表现在两个方面: `ssh 登录服务器`, 和`通过ssh 使用 git`, 报的错都是`Permissino denied (publickey)`

## git异常的解决

根据[github文档](https://help.github.com/articles/error-permission-denied-publickey/), 我可以解决第二个问题, 第一个问题(ssh login)解决后再来更新.

使用git出现问题, 并不局限于 github, 我连 gitlab 也是一样. 其实根据文档一项项检测,我在`ssh-add -l`这一句发现了问题, 虽然我生成了密钥, 但是它的输出显示我没有私钥, 照如下解决即可:

>**Tip**: On most systems the default private keys (~/.ssh/id_rsa
, ~/.ssh/id_dsa and ~/.ssh/identity) are **automatically** added to the SSH authentication agent. You shouldn't need to run `ssh-add path/to/key`
 unless you override the file name when you generate a key.

也就是说, 把你的id_rsa文件手动指定一下`ssh_add 文件路径`(不知道为什么会出这种鬼问题)

## SSH自动登录异常的解决
这是补充的内容, 刚刚解决.

首先, 你登不上这是服务端的问题, 没有把你的公钥写到`~/.ssh/authorized_keys`里面去, 你想办法把它写进去(当你 `ssh` 不上去的时候, `scp`当然也不行)

写进去后, 仍然报错, 我们继续看日志:
`ssh root@mydomain.com -v` 或 `ssh root@mydomain.com -vvv` (更详细的日志)

通过看日志, 你可以一步步看到问题:
```
...
debug2: service_accept: ssh-userauth
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug3: send packet: type 50
debug3: receive packet: type 51
debug1: Authentications that can continue: publickey
debug3: start over, passed a different list publickey
debug3: preferred publickey,keyboard-interactive,password
debug3: authmethod_lookup publickey
debug3: remaining preferred: keyboard-interactive,password
debug3: authmethod_is_enabled publickey
debug1: Next authentication method: publickey
debug1: Trying private key: .ssh/id_rsa
debug3: no such identity: .ssh/id_rsa: No such file or directory
debug2: we did not send a packet, disable method
debug1: No more authentication methods to try.
Permission denied (publickey).
```

注意截取的**倒数第4行**:
首先, 它昨天不是这么提示的, 它提示的是`.ssh/id_rsd`, 我到配置文件里把它改成了`rsa`, 因为我生成的是 `rsa`
> 配置文件的地址是: `/etc/ssh/ssh_config`, 改的节点名是`IdentityFile`

其次, 顺便看一下`RSAAuthentication` 和 `PubkeyAuthentication`这两项是不是 yes(如果是注释状态, 不要动, 默认是 yes)

这样, 服务端有你的公钥, 本地配置了 `IdentityFile` 路径, 就可以登录了, 但我一直没成功的原因在于, 我在原配置文件改的, 它是路径是`.ssh/id_rsa,.ssh/id_rsd`, 我单纯把 rsd 的去掉, 却没发现它的路径是错的, 直到看了日志提示这个文件不存在, 才想起把格式改对.

我碰到的情况不知道是不是个例, 比如同一个文件
```
UserKnownHostsFile ~/.ssh/known_hosts,~/.ssh/known_hosts2
```
这一行, 路径格式却是正确的, 匪夷所思.

总之, 你碰到`Permission denied (public key)`这个问题, 就结合 git 和 server 这两情况, 看是没有 identity, 还是 id_rsa路径配错了. (前提是公钥必须已经写到服务器上去了)
