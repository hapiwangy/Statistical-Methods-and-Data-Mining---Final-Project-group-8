import pandas as pd
import numpy as np
from scipy.stats import kruskal

data = pd.read_csv('data_binary_after_preprocessing.csv')
df = pd.DataFrame(data)

health_rating_no_diabetes = df[df['Diabetes_binary'] == 0]['GenHlth']
health_rating_diabetes = df[df['Diabetes_binary'] == 1]['GenHlth']

# Kruskal-Wallis_H檢驗
kruskal_test_stat, kruskal_test_p = kruskal(health_rating_no_diabetes, health_rating_diabetes)

print(f"Kruskal-Wallis Test Statistic: {kruskal_test_stat}")
print(f"Kruskal-Wallis Test p-value: {kruskal_test_p}")
