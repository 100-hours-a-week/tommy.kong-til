#1. 
# 가상의 데이터셋 생성
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

data = {
    '학생': ['A', 'B', 'C', 'D', 'E'],
    '수학': [90, np.nan, 85, 88, np.nan],
    '영어': [80, 78, np.nan, 90, 85],
    '과학': [np.nan, 89, 85, 92, 80]
}

df = pd.DataFrame(data)
print(df.info())
print(df.isna().sum())
for i in df.select_dtypes(include=np.number).columns:
    if df[i].isna().sum() > 0:
        df[i] = df[i].fillna(df[i].median())

for i in df.select_dtypes(include='object').columns:
    if df[i].isna().sum() > 0:
        df[i] = df[i].fillna(df[i].sort_values(ascending=False)[0])

for i in df.select_dtypes(include=np.number).columns:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]

df.reset_index(drop=True, inplace=True)

scaler = minmax_scale()

for i in df.select_dtypes(include=np.number).columns:
    df[i] = scaler.fit_transform(df[i])

print(df)