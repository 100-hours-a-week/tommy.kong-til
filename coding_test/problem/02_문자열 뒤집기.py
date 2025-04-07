# https://leetcode.com/problems/reverse-string/

# My Solution
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i = 0
        j = len(s) - 1
        while i < len(s):
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
            if i > j:
                break

# Solution 1 투포인터를 사용


class Solution:
    def reverseString(self, s: List[str]) -> None:
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

# Solution 2 파이썬다운 방식


class Solution:
    def reverseString(self, s: List[str]) -> None:
        s.reverse()

# Solution 2-1


class Solution:
    def reverseString(self, s: List[str]) -> None:
        s[:] = s[::-1]  # 공간 복잡도 제한을 피하는 방법
