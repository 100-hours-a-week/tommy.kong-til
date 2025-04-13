#
# @lc app=leetcode id=49 lang=python3
#
# [49] Group Anagrams
#

from collections import Counter

# @lc code=start

# my solution
# class Solution:
#     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
#         word_set = []
#         for i in strs:
#             for j in i:
#                 word = []
#                 word.append(j)
#                 word_set.append(word)
#         result = []
#         for i in range(len(word_set)):
#             for j in range(i, len((word_set))):
#                 word = []
#                 word.append(strs[i])
#                 if set(word_set[i]) == set(word_set[j]):
#                     word.append(strs[j])
#             result.append(word)
# return result


# book solution
import collections


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = collections.defaultdict(list)
        for word in strs:
            anagrams[''.join(sorted(word))].append(word)
        return list(anagrams.values())

# @lc code=end
