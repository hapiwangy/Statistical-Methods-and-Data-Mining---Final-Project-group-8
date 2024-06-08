import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import f
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal

# 讀取資料
data = pd.read_csv('/Users/j.c1.5/Desktop/資料採礦/FinalProject/data_binary_after_preprocessing.csv')

#過濾掉低於 Q1 - 1.5IQR 或高於 Q3 + 1.5IQR 的異常值
Q1 = data['BMI_afterNormalization'].quantile(0.25)
Q3 = data['BMI_afterNormalization'].quantile(0.75)
IQR = Q3 - Q1

data = data[(data['BMI_afterNormalization'] >= Q1 - 1.5 * IQR) & (data['BMI_afterNormalization'] <= Q3 + 1.5 * IQR)]

# 假設檢定
# 分成有糖尿病和無糖尿病的兩組
diabetes_group = data[data['Diabetes_binary'] == 1.0]['BMI_afterNormalization']
no_diabetes_group = data[data['Diabetes_binary'] == 0.0]['BMI_afterNormalization']

print("H0: Diabetes and non-diabetes groups 與 BMI 之間無顯著差異")
print("H1: Diabetes and non-diabetes groups 與 BMI 之間有顯著差異")

# 利用F-test檢驗變異數
var_diabetes_group = diabetes_group.var()
var_no_diabetes_group = no_diabetes_group.var()

f_stat = var_diabetes_group / var_no_diabetes_group

df1 = len(diabetes_group) - 1
df2 = len(no_diabetes_group) - 1
critical_value = f.ppf(0.9, df1, df2)

print("F統計量:", f_stat)
print("臨界值:", critical_value)

if f_stat > critical_value:
    print("結果顯示變異數存在顯著差異")
else:
    print("結果顯示變異數沒有顯著差異")

# # 進行獨立樣本t檢定
# t_stat, p_value = stats.ttest_ind(diabetes_group, no_diabetes_group)

#進行 Welch's t-test
t_stat, p_value = stats.ttest_ind(diabetes_group, no_diabetes_group, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# 判斷檢定結果
alpha = 0.05
if p_value < alpha:
    print("拒絕 H0: Diabetes and non-diabetes groups 與 BMI 之間有顯著差異")
else:
    print("無法拒絕 H0: Diabetes and non-diabetes groups 與 BMI 之間無顯著差異")
