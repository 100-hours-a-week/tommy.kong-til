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