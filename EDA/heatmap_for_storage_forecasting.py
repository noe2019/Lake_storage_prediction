# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 23:34:35 2018

@author: 20584059
"""

"""
Params: df - DataFrame of our Abalone data
Return: Generates a heatmap plot
"""#Create Correlation df
import pandas
import numpy as np 
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import math
#%matplotlib inline
dataframe1 = pandas.read_excel('correl.xlsx')
dataframe1.head()
dataset1 = dataframe1.values
np.shape(dataset1)

df2 = DataFrame(dataset1,columns = ['Storage',	'dswr', 'lftx', 'mlsp',	'p_f', 'p_u',	'p_v', 'p_z',	'p_th',	'p_zh',	'p5_f',	'p5_u',	'p5_v',	'p5_z',	'p5th',	'p5zh',	'p8_f',	'p8_u',	'p8_v',	'p8_z',	'p8th',	'p8_zh',	'p500',	'p850',	'pottmp', 'pr_wtr',	'prec',	'r500',	'r850',	'rhum',	'shum',	'temp'
],index= ['Storage',	'dswr', 'lftx', 'mlsp',	'p_f', 'p_u',	'p_v', 'p_z',	'p_th',	'p_zh',	'p5_f',	'p5_u',	'p5_v',	'p5_z',	'p5th',	'p5zh',	'p8_f',	'p8_u',	'p8_v',	'p8_z',	'p8th',	'p8_zh',	'p500',	'p850',	'pottmp', 'pr_wtr',	'prec',	'r500',	'r850',	'rhum',	'shum',	'temp'
])
fig, ax = plt.subplots(figsize=(25,15))
sns.heatmap(df2, ax=ax,annot=True,annot_kws={"size": 12})
plt.show()