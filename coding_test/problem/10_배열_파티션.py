# My solution 
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

# 오름차순 풀이
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        sum = 0
        pair = []
        nums.sort()
        for n in nums:
            # 앞에서부터 오름차순으로 페어를 만들어서 합 계산
            pair.append(n)
            if len(pair) == 2:
                sum += min(pair)
                pair = []
        return sum
    
# 짝수 번째 값 계산
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        sum = 0
        nums.sort()

        for i,n in enumerate(nums):
            # 짝수 번째 값의 합 계산
            if i % 2 == 0:
                sum += n

        return sum

# 파이썬 다운 방식 
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        return(sum(sorted(nums[::2])))
