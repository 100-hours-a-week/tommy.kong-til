#
# @lc app=leetcode id=561 lang=python3
#
# [561] Array Partition
#

# @lc code=start
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        results = []
        if n % 2 == 0:
         for i in range(0,len(nums),2):
            results.append(min(nums[i],nums[i+1]))
        else:
            for i in range(i+1,len(nums),2):
                results.append(min[i])
        return sum(results)
    

# @lc code=end

