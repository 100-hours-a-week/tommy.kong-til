class Solution:
    def isValid(self, s: str) -> bool:

        dic = {')': '(', ']': '[', '}': '{'}
        stack = []
        for i in s:
            if i not in dic:
                stack.append(i)
            elif not stack and dic[i] != stack.pop():
                return False
        return len(stack) == 0

# -----

# dic = {'(': ')', '[': ']', '{': '}'}

# print('(' in dic)  # True
# print(')' in dic)  # False

# 위 코드에서 '('은 dic의 키에 존재하므로 True를 반환하고, 
# ')'는 키에 없으므로 False를 반환합니다.

# 즉 키를 기준으로 in을 탐색한다. 


# -----

# 예제 2: s = "([)]"
# i = '(':

# i not in dic → True (여는 괄호).
# 스택에 추가: stack = ['('].
# [i = 'i = '[':

# i not in dic → True (여는 괄호).
# 스택에 추가: [stack = stack = ['(', '['].
# i = ')':

# i in dic → True (닫는 괄호).
# 스택이 비어 있지 않고(not stack → False), stack.pop() 결과 '['이 dic[')']와 다름.
# False 반환.

# 이런식으로 그전 걸 보는 거다.