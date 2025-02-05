# 2주차 주간 챌린지 brief code

"""
1. NumPy 배열 생성 및 연산

- NumPy를 사용해 3x3 정수 배열(1부터 9까지)과 2x5 실수 배열(0~1 사이 난수)을 생성하세요. 생성한 배열의 합계와 평균을 각각 계산하 고, 두 배열을 곱한 결과를 출력하세요.
"""

import numpy as np
array1 = np.arange(1,10).reshape(3,3)
array2 = np.random.random((2,5))
print(f"배열1 합계 : {np.sum(array1)}, 배열1 평균 : {np.mean(array1)}, 배열2 합계 : {np.sum(array2)}, 배열2 평균 : {np.mean(array2)}")
print()
# print(f"두 배열의 곱 : {np.dot(array1,array2)}") -> ValueError: operands could not be broadcast together with shapes (3,3) (2,5)
print()

"""
2. Pandas를 활용한 데이터 처리
- 아래 데이터를 DataFrame으로 생성하세요.
Name: [Alice, Bob, None, Charlie]
Age: [25, None, 28, 35]
City: [New York, None, Chicago, None]

- 누락된 이름은 "Unknown"으로, 나이는 평균 나이로 채우고 수정된 DataFrame을 출력하세요.
"""

import pandas as pd
data = {'Name': ['Alice', 'Bob', None, 'Charlie'],'Age': [25, None, 28, 35],'City': ['New York', None, 'Chicago', None]}
df = pd.DataFrame(data)
print(df)
print()
df.fillna({'Name':'Unknown', 'Age':df['Age'].mean().astype('int')}, inplace=True)
print(df)
print()

"""
3: HTTP 통신을 활용한 데이터 읽기 및 저장
- https://jsonplaceholder.typicode.com/todos에서 JSON 데이터를 가져오세요.
데이터를 파일로 저장한 뒤, "title" 키의 값을 출력하세요.
제출 방식: 1, 2, 3을 콜랩 페이지 하나로 하여 링크 제출
"""

import requests

url = 'https://jsonplaceholder.typicode.com/todos'
response = requests.get(url)
data = response.json()
print(data)
with open('todos.json', 'w') as f:
    f.write(str(data))