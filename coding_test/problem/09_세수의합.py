class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort() # 초반에 정렬을 해야지 쉽게 풀 수 있다. 
        
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue # continue를 하면 아래로 가는 게 아니라 위의 for loop으로 간다. 
            for j in range(i+1, len(nums)-1):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                for k in range(j+1, len(nums)):
                    if k > j+1 and nums[k] == nums[k-1]:
                        continue
                    if nums[i] + nums[j] + nums[k] == 0:
                        result.append([nums[i], nums[j], nums[k]])
                        
        return result


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        
        for i in range(len(nums) -2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum < 0:
                    left += 1
                elif sum >0:
                    right -= 1
                else:
                    result.append([nums[i],nums[left],nums[right]])
                    
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
        return result
                    # 이 값들로는 이미 정답을 만들었으므로 다음 조합을 위해 반드시 이동해야 합니다.
