---
layout: post
title: CocoaPods创建私有库过程拾遗
slug: cocoapods_private_library
date: 2022-02-27 17:44
status: publish
author: walker
categories: 
  - AI
tags:
  - cocoapods
  - 私有库
  - podspec
---

# 创建私有podspec

完整教程网上很多，我这里是曲曲折折弄好后的一些要点记录，里面的一些路径和库共同自[某篇教程](http://blog.wtlucky.com/blog/2015/02/26/create-private-podspec/)，可以直接看他们的教程。

想看极简的骨架过程可以参考我下面的笔记，当然肯定缺少很多细节，主要是记录一下核心思路，里面的一些库地址出于隐私我就使用了他们公布在网上的而不是自己的真实地址。

首先，涉及两个仓库，一个放代码，一个放spec，放spec的就是私有库

```bash
# 创建私有库 （就是host podspec文件的容器）
pod repo add WTSpecs https://coding.net/wtlucky/WTSpecs.git  #（这是spec仓库）

## 如果不是新建，删除和添加已有的语法：
pod repo remove WTSpecs
pod repo add WTSpecs git@coding.net:wtlucky/WTSpecs.git

# 创建pod lib（就是普通项目文件）
pod lib create podTestLibrary
### 可以选择尝试编辑一个组件放入Pod/Classes中，然后进入Example文件夹执行pod update命令，再打开项目工程可以看到，刚刚添加的组件已经在Pods子工程下

# 推送lib到remote
git add .
git commit -s -m "Initial Commit of Library"
git remote add origin git@coding.net:wtlucky/podTestLibrary.git  # 添加远端仓库（这是代码仓库）
git push origin master     # 提交到远端仓库

# 打rag，推tag
git tag -m "first release" 0.1.0
git push --tags     #推送tag到远端仓库

# 编辑podspec
### 请查阅相关字段文档，注意编辑tag号与你推的tag号一致
### 特别注意
### source_files(源码路径，一般在在libNmae/Classes/**/*), 
### resource_bundles(比如.bundle, .xcassets等)， 
### public_header_files(可以理解为Umbrella Header), 
### prefix_header_file(就是.pch文件)

# lint podspec（注意allow-warnings)
pod lib lint  --allow-warnings 
## 如果有私有源：
pod lib lint --sources='YourSource,https://mirrors.tuna.tsinghua.edu.cn/git/CocoaPods/Specs.git'
### 前面是私有源，逗号后是官方源，当然，因为我电脑用的是清华源，这里干脆也了往成一致了（不是必要）

# 如果不是用pod创建的项目，自行创建podspec文件：
 pod spec create PodTestLibrary git@coding.net:wtlucky/podTestLibrary.git  # 注意仓库名和仓库地址
```

本地测试podspec, in podfile:
```ruby
platform :ios, '9.0'

# 几种方式
pod 'PodTestLibrary', :path => '~/code/Cocoapods/podTest/PodTestLibrary'      # 指定路径
pod 'PodTestLibrary', :podspec => '~/code/Cocoapods/podTest/PodTestLibrary/PodTestLibrary.podspec'  # 指定podspec文件
```

```bash
# 向Spec Repo提交podspec(后面的参数是在消警告和错误的过程中加的，你可以尝试无参数先跑，碰到问题再逐个解决)
pod repo push WTSpecs PodTestLibrary.podspec --allow-warnings --use-libraries --skip-import-validation --verbose
### 完了后本地~/.cocoapods/repos和远端spec仓库都应该出现PodTextLibrary/0.1.0这个文件夹(对应你刚打的tag），里面有（且只有）刚才创建的podspec文件
```

使用
```ruby
pod 'PodTestLibrary', '~> 0.1.0'
```

* `--allow-warnings`, `--use-libraries`, `--skip-import-validation` 等参数灵活使用，目标就是为了通过验证

* `--no-clean` 可以在出错时打印更详细的信息（我加了`--verbose`后在build失败时会提示你加这个)

* 碰到有模块不支持i386什么的架构时，添加这个([更多看这篇文章](https://blog.nowcoder.net/n/68dac16078184973ac061027817a2d9a?from=nowcoder_improve))：

* ```rub
  s.xcconfig = {
      'VALID_ARCHS' =>  'x86_64 armv7 arm64',
    }
    s.pod_target_xcconfig = { 'ARCHS[sdk=iphonesimulator*]' => '$(ARCHS_STANDARD_64_BIT)' }
  ```

* `pod lint implicit declaration of function 'XXXX' is invalid in C99 [-Werror,-Wimplicit-function-declaration]` [看这里](https://blog.csdn.net/cnwyt/article/details/105073749) 

  * 很奇怪的问题，我前面的依赖确实添加了该宏定义的模块`s.dependency 'xxxx' 我目前是在问题文件里重新define一次这个宏解决的，

# podspec 进阶

```ruby
# [如果]每个子模块有自己的dependency, public headerfile, pchfile等
s.subspec 'NetWorkEngine' do |networkEngine|
    networkEngine.source_files = 'Pod/Classes/NetworkEngine/**/*'
    networkEngine.public_header_files = 'Pod/Classes/NetworkEngine/**/*.h'
    networkEngine.dependency 'AFNetworking', '~> 2.3'
end

s.subspec 'DataModel' do |dataModel|
    dataModel.source_files = 'Pod/Classes/DataModel/**/*'
    dataModel.public_header_files = 'Pod/Classes/DataModel/**/*.h'
end

s.subspec 'CommonTools' do |commonTools|
    commonTools.source_files = 'Pod/Classes/CommonTools/**/*'
    commonTools.public_header_files = 'Pod/Classes/CommonTools/**/*.h'
    commonTools.dependency 'OpenUDID', '~> 1.0.0'
end

s.subspec 'UIKitAddition' do |ui|
    ui.source_files = 'Pod/Classes/UIKitAddition/**/*'
    ui.public_header_files = 'Pod/Classes/UIKitAddition/**/*.h'
    ui.resource = "Pod/Assets/MLSUIKitResource.bundle"
    ui.dependency 'PodTestLibrary/CommonTools'
end
```

体现为：

```bash
$ pod search PodTestLibrary

-> PodTestLibrary (1.0.0)
   Just Testing.
   pod 'PodTestLibrary', '~> 1.0.0'
   - Homepage: https://coding.net/u/wtlucky/p/podTestLibrary
   - Source:   https://coding.net/wtlucky/podTestLibrary.git
   - Versions: 1.0.0, 0.1.0 [WTSpecs repo]
   - Sub specs:
     - PodTestLibrary/NetWorkEngine (1.0.0)
     - PodTestLibrary/DataModel (1.0.0)
     - PodTestLibrary/CommonTools (1.0.0)
     - PodTestLibrary/UIKitAddition (1.0.0)
```

使用：

```ruby
source 'https://github.com/CocoaPods/Specs.git'  # 官方库
source 'https://git.coding.net/wtlucky/WTSpecs.git'   # 私有库
platform :ios, '9.0'

pod 'PodTestLibrary/NetWorkEngine', '1.0.0'  #使用某一个部分
pod 'PodTestLibrary/UIKitAddition', '1.0.0'

pod 'PodTestLibrary', '1.0.0'   #使用整个库
```

