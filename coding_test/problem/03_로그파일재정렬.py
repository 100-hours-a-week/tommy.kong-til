# https://leetcode.com/problems/reorder-data-in-log-files/

from typing import List


class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letters, digits = [], []
        for log in logs:
            if log.split()[1].isdigit():
                digits.append(log)
            else:
                letters.append(log)

        # 두 개의 키를 람다 표현식으로 정렬
        letters.sort(key=lambda x: (x.split()[1:], x.split()[0]))
        # 리스트 메서드로 사용 (원본 리스트 변경)
        # list.sort(key=None, reverse=False)
        # key: 정렬 기준이 되는 함수를 지정합니다. 각 요소에 이 함수가 적용되고, 그 결과값을 기준으로 정렬합니다.
        # reverse: True면 내림차순, False면 오름차순으로 정렬합니다(기본값은 False).
        # 첫 번째 요소(로그 내용)를 주요 기준으로 삼고
        # 내용이 동일할 경우 두 번째 요소(식별자)로 정렬
        return letters + digits

# 람다 문법과 동일하다.
# def func(X):
#     return x.spltt()[1], X.spltt()[0]
# s.sort(key=func)
# s
