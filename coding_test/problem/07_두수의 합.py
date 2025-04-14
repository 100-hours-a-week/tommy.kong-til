# 내 솔루션 => 부르트 포스로 계산
# 시간 복잡도 O(n^2) 즉, 풀이는 가능하지만 비효율적이며 느리다.
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    return [i,j]

# in을 이용한 탐색 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, n in enumerate(nums):
            complement = target - n 
            if complement in nums[i+1:]:
                return [nums.index(n), nums[i+1:].index(complement) + (i+1)]

# 첫 번째 수를 뺀 결과 키 조회 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_map = {}
        for i, num in enumerate(nums):
            nums_map[num] = i
        
        for i, num in enumerate(nums):
            if target - num in nums_map and i != nums_map[target-num]:
                return [i, nums_map[target-num]]
# 타겟에서 첫번째 수를 빼면 두 번째 수를 바로 알아낼 수 있음. 
# 두번째 수를 키로 하고 기존의 인덱스는 값으로 바꿔서 딕셔너리로 저장해두면, 나중에 두번째 수를 키로 조회해서 정답을 즉시 찾아낼 수 있음 .
# 딕셔너리는 해시 테이블로 구현 

# 조회 구조 개선 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_map = {}
        for i, num in enumerate(num):
            if target - num in nums_map:
                return [nums_map[target-num],i]
        nums_map[num] = i
# 딕셔너리 저장과 조회를 2개의 for문으로 각각 처리했던 방식을 개선해서 이번에는 하나의 for로 합쳐서 처리 

# 투 포인터 이용 
# 투 포인터란 왼쪽 포인터와 오른쪽 포인터의 합이 타겟보다크면 오른쪽을 왼쪽으로 작다면 왼쪽을 오른쪽으로 옮기면서 값을 조정하는 방식 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums) - 1
        while not left == right:
            if nums[left] + nums[right] < target:
                left += 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                return [left, right]