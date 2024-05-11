#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：read_csv.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024-04-25 21:50 
'''

import pandas as pd
import numpy as np
from scipy.stats import yeojohnson


def drop_outer_data(df1, n_var):
    for label in list(df1.columns.values): # list(df1.columns.values)
        # std = np.squeeze(df1[[label]].std().to_numpy())
        # mean = np.squeeze(df1[[label]].mean().to_numpy())

        # 使用Box-Cox变换
        transformed_data, _ = yeojohnson(df1[label].dropna() + 1)  # 对数据进行Box-Cox变换
        transformed_mean = np.mean(transformed_data)  # 变换后的均值
        transformed_std = np.std(transformed_data)  # 变换后的标准差

        # transformed_df = pd.DataFrame(transformed_data, index=df1[label].dropna().index, columns=[label])

        # 删除异常值
        # df1 = df1[(df1[label] > transformed_mean - n_var * transformed_std) | (df1[label].isna())]
        # df1 = df1[(df1[label] < transformed_mean + n_var * transformed_std) | (df1[label].isna())]

        # 删除异常值
        df1 = df1[(df1[label] > (transformed_mean - n_var * transformed_std)) | (df1[label].notna())]
        df1 = df1[(df1[label] < (transformed_mean + n_var * transformed_std)) | (df1[label].notna())]

    return df1


df1 = pd.read_csv('G://data.csv')
df1 = df1.iloc[0:500, :]

# 对两列进行 Yeo-Johnson 变换，并将结果赋值给新的列
df1['卷绕正极极片张力最大值_transformed'], _ = yeojohnson(df1['卷绕正极极片张力最大值'] + 1)
df1['正极张力波动比例最大值_transformed'], _ = yeojohnson(df1['正极张力波动比例最大值'] + 1)

# 调用 drop_outer_data 函数删除异常值
df1 = drop_outer_data(df1, 15)

# 绘制数据分布
# plot_Normalize_distribution(df1, df_mean, df_std)
