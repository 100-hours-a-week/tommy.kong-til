# 01. 유효한 팰린드롬

# site : https://leetcode.com/problems/valid-palindrome/

# 주어진 문자열이 팰린드롬인지 확인하라. 대소문자를 구분하지 않으며, 영문자와 숫자만을 대상으로 한다

# My Solution
# 1. 문자열을 공백을 기준으로 나누고, 각 단어를 알파벳과 숫자만 남기고 소문자로 변환하여 리스트에 저장
# 2. 리스트를 뒤집어서 새로운 리스트에 저장
# 3. 두 리스트를 문자열로 변환하여 비교
# 4. 두 문자열이 같으면 True, 다르면 False 반환

import re
import collections
from collections import deque


class Solution:
    def isPalindrome(self, s: str) -> bool:
        string1 = []
        string2 = []
        for i in s.split():
            for j in i:
                if j.isalnum():
                    string1.append(j.lower())
        s1 = "".join(string1)
        for i in range(len(string1)):
            string2.append(string1.pop())
        s2 = "".join(string2)
        if s1 == s2:
            return True
        else:
            return False


# Solution 1 => 리스트로 변환
def isPalindrome(self, s: str) -> bool:
    strs = []
    for char in s:
        if char.isalnum():
            strs.append(char.lower())

        while len(strs) > 1:
            if strs.pop(0) != strs.pop():  # pop(0)을 지정하면 맨 앞의 값을 자져올 수 있다.
                return False
    return True


# Solution 2 => 데크 자료형을 이용한 최적화


def isPalindrome(self, s: str) -> bool:
    # deque()는 collections 모듈에서 제공하는 자료형으로, 양쪽 끝에서 삽입과 삭제가 가능한 큐를 구현할 수 있다.
    strs: Deque = collections.deque()
    for char in s:
        if char.isalnum():
            strs.append(char.lower())

    while len(strs) > 1:
        if strs.popleft() != strs.pop():
            return False
    return True


# Solution 3 => 슬라이싱을 이용한 최적화


def isPalindrome(self, s: str) -> bool:
    s = s.lower()
    s = re.sub('[^a-z0-9]', '', s)  # 정규 표현식을 사용하여 알파벳과 숫자만 남기고 나머지 문자를 제거
    return s == s[::-1]  # 슬라이싱을 이용하여 문자열을 뒤집는 방법
