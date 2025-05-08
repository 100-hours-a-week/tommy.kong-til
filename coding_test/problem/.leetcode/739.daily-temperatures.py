#
# @lc app=leetcode id=739 lang=python3
#
# [739] Daily Temperatures
#

# @lc code=start
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        result = []
        for i in range(len(temperatures)):
            for j in range(i+1, len(temperatures)):
                if temperatures[i] < temperatures[j]:
                    result.append(j-i)
                    break
                elif j == len(temperatures)-1:
                    result.append(0)
        result.append(0)
        return result

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:

        answer = [0] * len(temperatures)
        stack = []
        for i, cur in enumerate(temperatures):
            while stack and cur > temperatures[stack[-1]]:
                last = stack.pop()
                answer[last] = i - last
            stack.append(i)
        return answer

# 초기 상태:

# answer = [0, 0, 0, 0, 0, 0, 0, 0]
# stack = []
# 첫 번째 날 (i=0, cur=73):
# 스택이 비어 있으므로 stack = [0].

# 두 번째 날 (i=1, cur=74):
# 74 > 73 (스택의 마지막 인덱스 0의 온도).
# 스택에서 0을 꺼내고 answer[0] = 1 - 0 = 1.
# stack = [1].

# 세 번째 날 (i=2, cur=75):
# 75 > 74 (스택의 마지막 인덱스 1의 온도).
# 스택에서 1을 꺼내고 answer[1] = 2 - 1 = 1.
# stack = [2].

# 네 번째 날 (i=3, cur=71):
# 71 < 75 (스택의 마지막 인덱스 2의 온도).
# 스택에 3을 추가: stack = [2, 3].

# 다섯 번째 날 (i=4, cur=69):
# 69 < 71 (스택의 마지막 인덱스 3의 온도).
# 스택에 4를 추가: stack = [2, 3, 4].

# 여섯 번째 날 (i=5, cur=72):
# 72 > 69 (스택의 마지막 인덱스 4의 온도).
# 스택에서 4를 꺼내고 answer[4] = 5 - 4 = 1.
# 72 > 71 (스택의 마지막 인덱스 3의 온도).
# 스택에서 3을 꺼내고 answer[3] = 5 - 3 = 2.
# 스택에 5를 추가: stack = [2, 5].

# 일곱 번째 날 (i=6, cur=76):
# 76 > 72 (스택의 마지막 인덱스 5의 온도).
# 스택에서 5를 꺼내고 answer[5] = 6 - 5 = 1.
# 76 > 75 (스택의 마지막 인덱스 2의 온도).
# 스택에서 2를 꺼내고 answer[2] = 6 - 2 = 4.
# 스택에 6을 추가: stack = [6].

# 여덟 번째 날 (i=7, cur=73):
# 73 < 76 (스택의 마지막 인덱스 6의 온도).
# 스택에 7을 추가: stack = [6, 7].
# 반복 종료:
# 스택에 남아 있는 인덱스(6, 7)는 더 따뜻한 날이 없으므로 answer 값은 그대로 유지.

# 최종 결과
# 요약
# 스택을 사용하여 현재 온도와 이전 온도를 비교하며 더 따뜻한 날을 찾습니다.
# 시간 복잡도는 **O(n)**으로, 각 온도는 스택에 한 번 추가되고 한 번 제거됩니다.
# 공간 복잡도는 스택에 저장된 인덱스 수에 비례합니다.

# @lc code=end
