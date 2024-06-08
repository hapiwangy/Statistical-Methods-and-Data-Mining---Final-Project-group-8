# %%
# 先對資料做讀取
import pandas as pd
# 選擇要對哪個檔案進行操作
df = pd.read_csv(r"C:\Users\ASUS\Desktop\大四下課程\final_project\diabetes_012_health_indicators_BRFSS2015.csv")
# df = pd.read_csv(r"C:\Users\ASUS\Desktop\大四下課程\final_project\diabetes_binary_health_indicators_BRFSS2015.csv")
# 照這個步驟可以得知我們的資料都是沒有缺失值
print(df.info())
# %%
# 經過檢查以及比對，得知道有部分資料屬於數值類的資料，因此對他們做normalization
# 由於其他欄位的值屬於nomial 或是 ordinal，因此判斷並不適合做normailzation
# 以下提供適合做normalization的欄位。
# 'BMI','MentHlth','PhysHlth','Income', 'Education', 'Age', 'GenHlth'
titles = df.keys()
for title in titles:
    print('---front---')
    print(df[title].describe())
    print("---end---")
# %%
# 針對上面篩選出來的欄位做normalization
# from sklearn.preprocessing import Normalizer
# 發現使用normalization效果不好，因此轉用standardlizer
from sklearn.preprocessing import StandardScaler
# normalizer = Normalizer()
stand = StandardScaler()
for title in ['BMI','MentHlth','PhysHlth','Income', 'Education', 'Age', 'GenHlth']:
    df[title + f'_afterNormalization'] = stand.fit_transform(df[[title]])
# df[f'BMI_afterNormalization'] = stand.fit_transform(df[['BMI']])

# %%
# 檢視結果
df.describe()
# 根據檔案名稱要記得更改
df.to_csv('data_012_after_preprocessing.csv')
# %%
