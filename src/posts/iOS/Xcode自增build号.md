---
layout: post
title: Xcode自增build号
slug: Xcode自增build号
date: 2022-01-14 00:00
status: publish
author: walker
categories: 
  - iOS
---

脚本:
```bash
buildNumber=$(/usr/libexec/PlistBuddy -c "Print CFBundleVersion" "${PROJECT_DIR}/${INFOPLIST_FILE}")  
buildNumber=$(($buildNumber + 1))  
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $buildNumber" "${PROJECT_DIR}/${INFOPLIST_FILE}"
```
To use this script, follow these steps:
1. Select your app target in the project overview.
2. Select **Build Phases**.
3. Add a new build phase ("**New Run Script Phase**").
4. Enter the above script in the appropriate field.
5. In your `Info.plist`, ensure the current build number is numeric (是的, 主要是保证你原来填写的确实是数字就行了)


![](../assets/1859625-2618b44e2e4cdcd9.png)

来源: http://crunchybagel.com/auto-incrementing-build-numbers-in-xcode/

或者一些别的思考: 
http://stackoverflow.com/questions/9258344/better-way-of-incrementing-build-number
