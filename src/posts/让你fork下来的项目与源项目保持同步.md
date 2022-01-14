---
layout: post
title: 让你fork下来的项目与源项目保持同步
slug: 让你fork下来的项目与源项目保持同步
date: 2019-03-23 00:00
status: publish
author: walker
categories: 
  - Skill
tags:
  - git
  - github
  - fork
  - upstream
---

原文[在此](https://2buntu.com/articles/1459/keeping-your-forked-repo-synced-with-the-upstream-source/), 建议阅读, 我把关键步骤抽出来了, 方便概览

(也就是add remote upstream, fetch upstream, rebase, 再push)

Step 1: Forking a repo
```
git clone https://github.com/nitin-test/blog-example-fork.git
git remote add upstream https://github.com/nitstorm/blog-example.git
git remote
git remote show origin
git remote show upstream
```
Step 2: Making changes and submitting Pull Requests
```
git checkout -b word-addition
git commit -am "Adds the word memory"
git push origin word-addition
# 网页端发起merge到master
```
Step 3: Keeping the forked repo synced with the main repo
```
# 确保你是在master分支
git checkout master
git fetch upstream
git rebase upstream/master
git push origin master
```
