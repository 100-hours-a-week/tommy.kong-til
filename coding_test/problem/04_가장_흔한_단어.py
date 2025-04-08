# https://leetcode.com/problems/most-common-word/

# 금지된 단어를 제외한 가장 흔하게 등장하는 단어를 출력하라

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        for i in range(len(paragraph)):
