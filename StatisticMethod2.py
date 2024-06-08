import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import f
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import wilcoxon
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from decimal import Decimal

# 讀取資料
data = pd.read_csv('data_binary_after_preprocessing.csv')
df = pd.DataFrame(data)

def chi2_test(var1, var2, data):
    contingency_table = pd.crosstab(data[var1], data[var2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p

variables = ['Stroke', 'HvyAlcoholConsump', 'HighChol', 'HighBP', 'HeartDiseaseorAttack', 'Sex']
results = {}

# Chi-square Test
for var in variables:
    chi2, p = chi2_test(var, 'Diabetes_binary', df)
    results[var] = {'chi2': chi2, 'p-value': p}

for var, result in results.items():
    print(f"Chi-square test for {var} and Diabetes:")
    print(f"Chi-square statistic: {result['chi2']}")
    print(f"p-value: {result['p-value']}\n")

# ANOVA
model = ols('BMI ~ C(Diabetes_binary)', data=df).fit()
anova_results = anova_lm(model)

print("ANOVA results for BMI and Diabetes:")
print(anova_results)

# Sign Test
age_diabetes = df[df['Diabetes_binary'] == 1]['Age']
age_no_diabetes = df[df['Diabetes_binary'] == 0]['Age']

#檢查數據長度是否一致
if len(age_diabetes) != len(age_no_diabetes):
    min_length = min(len(age_diabetes), len(age_no_diabetes))
    age_diabetes = age_diabetes[:min_length]
    age_no_diabetes = age_no_diabetes[:min_length]

sign_test_stat, sign_test_p = wilcoxon(age_diabetes, age_no_diabetes)

print(f"Sign Test Statistic: {sign_test_stat}")
print(f"Sign Test p-value: {sign_test_p}")
