# 차원 - 미니퀘스트
## 1.
import numpy as np

array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(array.ndim)
print()
## 2.
array = np.array([10, 20, 30, 40, 50, 60])
print(array.reshape((2, 3)))
print()
## 3.
array = np.array([7, 14, 21])
print(array[:, np.newaxis].ndim)
print()
# 형태
## 1.
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array.shape)
print()
## 2.
array = np.array([10, 20, 30, 40, 50, 60])
array = array.reshape((2, 3))
print(array.shape)
print()
## 3.
array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
print(array.reshape((3, 2, 2)).shape)
print()
# 데이터 타입
## 1.
array = np.array([10, 20, 30])
print(array.dtype)
print()
## 2.
array = np.array([1, 2, 3], dtype="int64")
array = array.astype("float64")
print(array.dtype)
print()
## 3.
array = np.array([100, 200, 300])
array = array.astype("uint8")
print(array.nbytes)
print()
# 인덱스
## 1.
array = np.array([10, 20, 30, 40, 50])
print(array[-1])
print()
## 2.
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
print(matrix[0, :])
print(matrix[:, 1])
print()
## 3.
array = np.array([5, 15, 8, 20, 3, 12])
print(array[array > 10])
print(np.where(array > 10))
print()
# 연산
## 1.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
print()
## 2.
matrix = np.array([[10, 20, 30], [40, 50, 60]])
vector = np.array([1, 2, 3])
print(np.add(matrix, vector))
print()
## 3.
array = np.array([[3, 7, 2], [8, 4, 6]])
print(np.max(array, axis=0))
print(np.max(array, axis=1))
print()
# 유니버셜 함수
## 1.
array = np.array([1, 2, 3, 4])
print(np.power(array, 2))
print()
## 2.
array1 = np.array([10, 20, 30])
array2 = np.array([1, 2, 3])
array1 += array2
print(array1)
print()
## 3.
array = np.array([1, np.e, 10, 100])
print(np.log(array)[np.log(array) > 1])
print()
# Series
## 1.
import pandas as pd

print(pd.Series([5, 10, 15, 20]).index)
print()
## 2.
data = {"a": 100, "b": 200, "c": 300}
series = pd.Series(data)
print(series.loc["b"])
print(series.get("b"))
print()
## 3.
series = pd.Series([1, 2, None, 4, None, 6])
print(series.isna())
series.fillna(0, inplace=True)
print(series)
print()
# 데이터 프레임
## 1.
data = {"이름": ["홍길동", "김철수", "박영희"], "나이": [25, 30, 28], "성별": ["남", "남", "여"]}
df = pd.DataFrame(data)
print(df.columns)
print()
## 2.
data = {"이름": ["홍길동", "김철수", "박영희"], "나이": [25, 30, 28], "성별": ["남", "남", "여"]}
df = pd.DataFrame(data)
df.sort_values(by="나이", ascending=False, inplace=True)
print(df)
print()
## 3.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신"],
    "국어": [85, 90, 88, 92],
    "영어": [78, 85, 89, 87],
    "수학": [92, 88, 84, 90],
}
df = pd.DataFrame(data)
df["총점"] = df["국어"] + df["영어"] + df["수학"]
print(df[df["총점"] >= 250])
print()
# 필터링
## 1.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신", "강감찬"],
    "나이": [25, 30, 35, 40, 45],
    "도시": ["서울", "부산", "서울", "대구", "부산"],
}

df = pd.DataFrame(data)
print(df[df["나이"] >= 30])
print()
## 2.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신", "강감찬"],
    "나이": [25, 30, 35, 40, 45],
    "도시": ["서울", "부산", "서울", "대구", "부산"],
    "점수": [85, 90, 75, 95, 80],
}

df = pd.DataFrame(data)
print(df[(df["점수"] >= 80) | (df["도시"] == "서울")])
print()

## 3.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신", "강감찬"],
    "나이": [25, 30, 35, 40, 45],
    "도시": ["서울", "부산", "서울", "대구", "부산"],
    "점수": [85, 90, 75, 95, 80],
}

df = pd.DataFrame(data)
print(df.query("나이 >= 35 and 점수 > 80"))
print()

# 그룹화
## 1.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신"],
    "부서": ["영업", "영업", "인사", "인사"],
    "급여": [5000, 5500, 4800, 5100],
}

df = pd.DataFrame(data)
print(df.groupby("부서").sum())
print()
## 2.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신", "강감찬", "신사임당"],
    "부서": ["영업", "영업", "인사", "인사", "IT", "IT"],
    "급여": [5000, 5500, 4800, 5100, 6000, 6200],
}
df = pd.DataFrame(data)
print(df.groupby("부서").agg({"급여": ["sum", "mean"]}))
print()
## 3.
data = {
    "이름": ["홍길동", "김철수", "박영희", "이순신", "강감찬", "신사임당"],
    "부서": ["영업", "영업", "인사", "인사", "IT", "IT"],
    "급여": [5000, 5500, 4800, 5100, 6000, 6200],
}

df = pd.DataFrame(data)
print(df.groupby("부서").filter(lambda x: x["급여"].mean() > 5000))
print()
# 병합
# 1.
df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [10000, 20000, 30000]})
df3 = pd.merge(df1,df2,on='고객ID',how='inner')
print(df3)
print()

# 2. 
df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [15000, 25000, 35000]})
df3 = pd.merge(df1,df2,on='고객ID',how='left')
print(df3)
print()
# 3.
df1 = pd.DataFrame({
    '고객ID': [1, 2, 3],
    '도시': ['서울', '부산', '대전'],
    '구매액': [10000, 20000, 30000]
})

df2 = pd.DataFrame({
    '고객ID': [1, 2, 3],
    '도시': ['서울', '부산', '광주'],
    '구매액': [15000, 25000, 35000]
})

df3 = pd.merge(df1,df2,on=['고객ID','도시'],suffixes=['_1','_2'])

print(df3)
print()

# 결측치 처리
## 1.
data = {'이름': ['홍길동', '김철수', np.nan, '이영희'],
        '나이': [25, np.nan, 30, 28],
        '성별': ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)
print(df.isna().sum())
print()

## 2.
data = {'이름': ['홍길동', '김철수', np.nan, '이영희'],
        '나이': [25, np.nan, 30, 28],
        '성별': ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)
print(df.dropna())
print()

## 3.
data = {'이름': ['홍길동', '김철수', np.nan, '이영희'],
        '나이': [25, np.nan, 30, 28],
        '성별': ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)
df['나이'] = df['나이'].fillna(df['나이'].mean()).astype('int')
print(df)
print()

# 피벗
## 1. 
data = {
    '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250]
}

df = pd.DataFrame(data)

df_new = df.pivot(index='날짜',columns='제품',values='판매량')
print(df_new)
print()

#@ 2.
data = {
    '카테고리': ['전자', '가전', '전자', '가전'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250]
}

df = pd.DataFrame(data)
print(df.pivot_table(index='카테고리',columns='제품',values='판매량',aggfunc='sum'))
print()

## 3.
data = {
    '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250],
    '이익': [20, 50, 30, 60]
}

df = pd.DataFrame(data)

print(df.pivot(index='날짜',columns='제품',values=['판매량','이익']))
print()

# 중복 제거
## 1.
data = {
    '이름': ['김철수', '이영희', '김철수', '박민수'],
    '나이': [25, 30, 25, 40],
    '성별': ['남', '여', '남', '남']
}

df = pd.DataFrame(data)

print(df.drop_duplicates())
print()

## 2.
data = {
    '제품': ['노트북', '태블릿', '노트북', '스마트폰'],
    '가격': [1500000, 800000, 1500000, 1000000],
    '카테고리': ['전자기기', '전자기기', '전자기기', '전자기기']
}

df = pd.DataFrame(data)

df.drop_duplicates(subset=['제품'],inplace=True)
print('없음' if df.duplicated().sum()  == 0 else '있음')
print()

## 3.
data = {
    '학생': ['김민수', '박지현', '김민수', '이정훈'],
    '성적': [90, 85, 90, 88],
    '학교': ['A고', 'B고', 'A고', 'C고']
}

df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
print(df)
print()

# 문자열 처리 

## 1.
data = pd.Series(["HELLO", "WORLD", "PYTHON", "PANDAS"])
print(data.str.lower())
print()

## 2.
df = pd.DataFrame({"이름": [" John Doe ", "Alice ", " Bob", "Charlie Doe "]})
print(df['이름'][df['이름'].str.strip().str.contains('Doe')])
print()

## 3.
df = pd.DataFrame({"설명": ["빅데이터 분석", "데이터 과학", "머신 러닝", "딥 러닝"]})
def abbre(x):
    sp = x.split(' ')
    return sp[0][0] + sp[1][0]
df['약어'] = df['설명'].apply(abbre)
print(df['약어'])