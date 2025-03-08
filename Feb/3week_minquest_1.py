# 3week_minquest
# 정형데이터
# 1.
import pandas as pd

data = {
    "이름": ["김철수", "이영희", "박민수", "최지현", "홍길동"],
    "나이": [25, 30, 35, 28, 40],
    "직업": ["개발자", "마케터", "개발자", "디자이너", "CEO"],
    "연봉": [4000, 3500, 5000, 4200, 10000],
    "가입일": ["2020-05-21", "2019-07-15", "2021-01-10", "2018-11-03", "2017-09-27"],
}
print(pd.DataFrame(data).info())
print()
# 2.
data = {
    "이름": ["김철수", "이영희", "박민수", "최지현", "홍길동", "정지훈", "이지은"],
    "나이": [25, 30, 35, 28, 40, 50, 22],
    "직업": ["개발자", "마케터", "개발자", "디자이너", "CEO", "디자이너", "마케터"],
    "연봉": [4000, 3500, 5000, 4200, 10000, 4600, 3300],
    "가입일": [
        "2020-05-21",
        "2019-07-15",
        "2021-01-10",
        "2018-11-03",
        "2017-09-27",
        "2016-04-11",
        "2022-03-19",
    ],
}
df = pd.DataFrame(data)
print(df[(df["나이"] >= 30) & (df["연봉"] <= 5000)])
print()
# 3.
data = {
    "이름": ["김철수", "이영희", "박민수", "최지현", "홍길동", "정지훈", "이지은"],
    "나이": [25, 30, 35, 28, 40, 50, 22],
    "직업": ["개발자", "마케터", "개발자", "디자이너", "CEO", "디자이너", "마케터"],
    "연봉": [4000, 3500, 5000, 4200, 10000, 4600, 3300],
    "가입일": [
        "2020-05-21",
        "2019-07-15",
        "2021-01-10",
        "2018-11-03",
        "2017-09-27",
        "2016-04-11",
        "2022-03-19",
    ],
}
df = pd.DataFrame(data)
df["가입일"] = pd.to_datetime(df["가입일"])
df["연봉"] = df["연봉"].where(df["가입일"].dt.year > 2019, df["연봉"] * 1.1)
print(df["연봉"].mean())
print()
# 비정형데이터
# 1.
import json

data = """
[
    {"이름": "김철수", "나이": 25, "직업": "개발자", "연봉": 4000},
    {"이름": "이영희", "나이": 30, "직업": "마케터", "연봉": 3500},
    {"이름": "박민수", "나이": 35, "직업": "디자이너", "연봉": 4200}
]
"""
data = json.loads(data)
df = pd.DataFrame(data)
print(df)
print()
# 2.
import re

text = "안녕하세요!!! 저는 AI 모델-입니다. 12345 데이터를   정리해 보겠습니다."
print(re.sub(r"\s+", " ", "".join(re.findall(r"[가-힣]|\s+", text))))
print()
# 3.
text = "자연어 처리는 재미있다. 파이썬과 pandas를 활용하면 편리하다. 데이터 분석은 흥미롭다."
word_leng = [(i, len(i)) for i in text.split()]
df = pd.DataFrame(word_leng, columns=["문장", "단어 개수"])
print(df)
print()
# 막대 그래프
import matplotlib.pyplot as plt

# 1.
categories = ["A", "B", "C", "D", "E"]
values = [12, 25, 18, 30, 22]
df = pd.DataFrame(zip(categories, values), columns=["categories", "values"])
df.plot.bar()
plt.show()
# 2.
categories = ["A", "B", "C", "D", "E"]
values_2023 = [10, 15, 20, 25, 30]
values_2024 = [5, 10, 12, 18, 22]
df = pd.DataFrame(
    zip(categories, values_2023, values_2024),
    columns=["categories", "values_2023", "values_2024"],
)
df.plot.bar(stacked=True)
plt.show()

# 3.
import numpy as np

departments = ["Sales", "Marketing", "IT", "HR", "Finance"]
performance_2023 = [80, 70, 90, 60, 75]
performance_2024 = [85, 75, 95, 65, 80]
bar_width = 0.4
x = np.arange(len(departments))
plt.bar(x - bar_width / 2, performance_2023, width=bar_width, label="2023")
plt.bar(x + bar_width / 2, performance_2024, width=bar_width, label="2024")
plt.xticks(x, departments)
plt.legend()
plt.show()
