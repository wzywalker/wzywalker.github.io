# -*- coding: utf-8 -*-
"""Sample Configuration
"""

source_dir = "../src/"
build_dir = "../dist/"
# template = {
#     "name": "Galileo",
#     "type": "local",
#     "path": "../Galileo"
# }
template = {
    "name": "Prism",
    "type": "git",
    "url": "https://github.com/Reedo0910/Maverick-Theme-Prism.git",
    "branch": "deploy"
}

# For Maverick
site_prefix = "/"
# template = "Galileo"
index_page_size = 10
archives_page_size = 30
fetch_remote_imgs = False
enable_jsdelivr = {
    "enabled": True,
    "repo": "wzywalker/wzywalker.github.io@main"
}
locale = "Asia/Shanghai"
category_by_folder = True

# For site
site_name = "walker's code blog"
site_logo = "${static_prefix}android-chrome-512x512.png"
site_build_date = "2019-12-06T12:00+08:00"
author = "walker"
email = "walker.wzy@gmail.com"
author_homepage = "https://wzy.one"
description = "coder, reader"
key_words = ["Maverick", "AlanDecode", "Galileo", "blog"]
language = 'english'
external_links = [
    {
        "name": "Maverick",
        "url": "https://github.com/AlanDecode/Maverick",
        "brief": "🏄‍ Go My Own Way."
    },
    {
        "name": "Triple NULL",
        "url": "https://www.imalan.cn",
        "brief": "Home page for AlanDecode."
    }
]
nav = [
    {
        "name": "Home",
        "url": "${site_prefix}",
        "target": "_self"
    },
    {
        "name": "Archives",
        "url": "${site_prefix}archives/",
        "target": "_self"
    },
    {
        "name": "About",
        "url": "${site_prefix}about/",
        "target": "_self"
    }
]

social_links = [
    {
        "name": "Twitter",
        "url": "https://twitter.com/walkerwzy",
        "icon": "gi gi-twitter"
    },
    {
        "name": "GitHub",
        "url": "https://github.com/walkerwzy",
        "icon": "gi gi-github"
    },
    {
        "name": "Weibo",
        "url": "https://weibo.com/1071696872",
        "icon": "gi gi-weibo"
    }
]

valine = {
    "enable": True,
    "el": '#vcomments',
    "appId": "7tP92LoqK2cggW61DvJmWBo0-gzGzoHsz",
    "appKey": "iQCtrtlr8eKrQllM03GMESMJ",
    "visitor": True,
    "recordIP": True
}

head_addon = ''

footer_addon = ''

body_addon = ''
