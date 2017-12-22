# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib notebook
# 解决matplotlib显示中文问题
# 仅适用于Windows
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

report_2016_datafile=r'd:\Data\2016.csv'
report_2015_datafile=r'd:\Data\2015.csv'

report_2015_data=pd.read_csv(report_2015_datafile)
report_2016_data=pd.read_csv(report_2016_datafile)
print('2015年数据概览：')
report_2015_data.info()
report_2015_data.describe()

print('2016年数据概览：')
report_2016_data.info()
report_2016_data.describe()
print('2016的数据预览：')
print(report_2016_data.head())
print('2015年报告，前10条记录幸福指数：\n',report_2015_data['Happiness Score'][:10])

print('2016年报告，前10条记录幸福指数：\n',report_2016_data['Happiness Score'][:10])
# report_2016_data['Happiness Score'].plot(kind='hist',alpha=0.7)
fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,sharex=True)
#幸福指数的直方图
ax=plt.gca()
ax.axis([2,8,2,30])
ax1.hist(report_2016_data['Happiness Score'],alpha=0.7,bins=10,label='2016幸福指数')
ax2.hist(report_2015_data['Happiness Score'],alpha=0.7,bins=10,label='2015幸福指数')
ax1.legend()
ax2.legend()
plt.show()
report_2016=report_2016_data.copy()
grouped=report_2016.groupby('Region')
print('\n2016年幸福指数报告：')
for group,frame in grouped:
    print('{}：最大值{:.2f}，最小值{:.2f}，平均值{:.2f}，中间值{:.2f}'.format(group,frame['Happiness Score'].max(),
                                             frame['Happiness Score'].min(),
                                             frame['Happiness Score'].mean(),
                                             frame['Happiness Score'].median(),))
report_2015_data.set_index('Country',inplace=True)
report_2016_data.set_index('Country',inplace=True)
print('2015年，数据预览：')
print(report_2015_data['Happiness Rank'])
print('2016年，数据预览：')
print(report_2016_data['Happiness Rank'])

report_2015_data.sort_index(inplace=True)
report_2016_data.sort_index(inplace=True)
print('2015-2016排名变化：')
change=report_2015_data['Happiness Rank']-report_2016_data['Happiness Rank']
print(change)
# se2015=pd.Series(report_2015_data['Happiness Rank'],index=report_2015_data.index)
# se2016=pd.Series(report_2016_data['Happiness Rank'],index=report_2016_data.index)
# change=se2015-se2016
# print(change)
print('中国的排名变化：')
print(change['China'])
print('2015-2016幸福指数上升最快的国家', change.argmax())
# 查看下降最快的国家
print('2015-2016幸福指数下降最快的国家', change.argmin())