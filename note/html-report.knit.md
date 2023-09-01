---
title: "Health statics"
date: "2023-08-30"
bibliography: refer.bib
csl: nature-methods.csl
link-citations: true
nocite: "@*"
output: 
  bookdown::html_document2:
    toc: true
    fig_caption: true 
    css: 'font.css'
---






# 统计描述

  首先了解数据的频数分布，选用合适的集中以及离散趋势描述数据。

## 数值变量

频数分布图形： range / n 取整数。

- 正态分布数据选用算术均值，方差/标准差/变异系数描述其分布。 
   
  理论公式：  
  计算公式：

> 标准差，变异系数是同单位的。变异系数用于不同尺度，均值相差较大的数据。
> 方差/标准差/变异系数的计算都依赖于均值的计算。

- 对数正态分布数据选用几何均值，全距/四分位数。  
  
  理论公式：  
  计算公式：
- 任意其它分布选用中位数，全距/四分位数

## 分类变量

  分类变量数据主要依赖于各种相对数描述，包括proportion, rate, ration.其中rate主要涉及到时间的概念。

> 数据的标准化法
> 直接化法，按总人口统一人口数。
> 间接法， 每组死亡人数与预期死亡人数之比互相比较。
>
> 动态数列
> 定基/环比，变化/增长（-1），平均发展速度，平均变化速度。

