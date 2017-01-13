---
layout:     post
title:      "Welcome to Hann's Blog"
subtitle:   "Hello World!"
date:       2017-01-08
author:     "Hann"
header-img: "img/post-bg-2015.jpg"
tags:
    - Life
---

> “Yeah, It's on. ”


## 前言

Hann's Blog 就这么开通了。

[跳过废话，直接看技术实现 ](#build)

2016年暑假开始，我来到了Ninebot & Segway Robotics实习，通过推研面试后又进入了THU CV-AI LAB，就这样我顺利跳进了计算机视觉的大坑。


身边的大牛总喜欢写写文章教导一下小学弟(xiao xue mei)，作为一个程序猿， 我也尝试在CSDN上写写博客，然而发现CSDN的公式渲染实在太渣，模板也难看得一逼，索性像泽贤一样把博客挂在Github上好了。

当我注册好域名hann.wang才发现，原来Github Pages的自定义域名是没法弄证书的，这是信仰HTTPS Everythere的我不能忍的啊。后来在github-tools/github的issues中发现有人提到了Netlify，试一试果然很NB，于是博客的事就这么定下来了。


<p id = "build"></p>
---

## 正文

简单说说搭建这个博客的过程。  

Google一下（就不用某度）应该能找到很多 [GitHub Pages](https://pages.github.com/) + [Jekyll](http://jekyllrb.com/) 快速搭建Blog的技术方案，不过更简单的配置方法是——fork别人的Blog (^_^)，然后开始写Markdown就可以了~

Jekyll的本地调试需要用到ruby，先安装ruby，然后使用TUNA提供的[RubyGem源](https://mirrors.tuna.tsinghua.edu.cn/rubygems/)安装bundler，在仓库根目录下执行bundle install即可（前提是配置好Gemfile哈）。

编译网站：
```
bundle exec jekyll build
```

运行网站：
```
bundle exec jekyll serve --no-watch
```

在Github上部署个人主页时，需要建立一个名称为`username.github.io`的仓库，然后将网站部署到master分支中。如果需要绑定自定义域名，可以在仓库根目录中建立`CNAME`文件，填写域名地址。不过自定义域名使用https访问时会提示证书错误，目前Github无法给自定义域名设置SSL证书。

[Netlify](https://www.netlify.com)提供了绝佳的解决方案，从Github仓库自动部署，可以免费为自定义域名设置Let's Encrypt提供的SSL证书。使用Github登陆并选择自己的博客仓库就可以了，以后每次push操作都会自动触发Netlify重新编译部署。


## 后记

由于毕设和实习工作繁忙，不知道自己能否有足够的精力维护好这个博客。不过我还是很希望能时常做点总结，未来的某一天回头看看自己走过的路，或许还是一件很有意思的事。


—— Hann @ Jan. 8, 2017.
