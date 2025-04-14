#
# @lc app=leetcode id=42 lang=python3
#
# [42] Trapping Rain Water
#

# @lc code=start
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        total_water = 0
        n = len(height)

        # 각 위치에서 담을 수 있는 물의 양 계산
        for i in range(1, n-1):
            # 현재 위치의 왼쪽에서 가장 높은 막대 찾기
            left_max = max(height[:i])

            # 현재 위치의 오른쪽에서 가장 높은 막대 찾기
            right_max = max(height[i+1:])

            # 현재 위치에 담길 수 있는 물의 높이 계산
            min_height = min(left_max, right_max)

            # 물이 담기려면 현재 높이보다 좌우 최대 높이의 최소값이 커야 함
            if min_height > height[i]:
                total_water += min_height - height[i]

        return total_water

# @lc code=end
