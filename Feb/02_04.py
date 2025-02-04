import numpy as np

print(np.__version__)

a = np.array([1, 2, 3, 4, 5])  # 함수호출이 있다

print(a)
print(type(a))

array = np.array([1, 2, 3, 4, 5])
print(array.dtype)

int_array = np.array([1, 2, 3, 4, 5], dtype='int32')
print(int_array.dtype)

float_array = np.array([1, 2, 3, 4, 5], dtype='float32')
print(float_array.dtype)

string_array = np.array(['1', '2', '3', '4', '5'], dtype='str')
print(string_array.dtype)

converted_int_array = float_array.astype('int32')
print(converted_int_array.dtype)

array = np.array([1, 2, 3, 4, 5])
print(array[-1])


matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([1, 2, 3])
print(matrix + vector)

a = np.array([1, 2, 3 ])
result = np.empty_like(a)
print(result)
np.multiply(a, 2, out=result)
print(result)

import pandas as pd

data = {'날짜' : ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02', '2021-01-03'],
        '제품' : ['TV', '냉장고', '세탁기', '냉장고', 'TV'],
        '판매량' : [100, 200, 300, 250, 150]}

df = pd.DataFrame(data)

print(df)

df_pivot = df.pivot(index='날짜', columns='제품', values='판매량')
print(df_pivot)