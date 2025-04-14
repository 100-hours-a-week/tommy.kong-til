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

# 투포인터 이용


class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        volume = 0
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]

        while left < right:
            left_max, right_max = max(height[left], left_max), max(
                height[right], right_max)
            # 더 높은 쪽을 향해 투 포인터 이동
            if left_max <= right_max:
                volume += left_max - height[left]
                left += 1
            else:
                volume += right_max - height[right]
                right -= 1
        return volume

# 스텍 이용


class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        volume = 0

        for i in range(len(height)):
            # 변곡점을 만나는 경우
            while stack and height[i] > height[stack[-1]]:
                # 스택에서 꺼낸다
                top = stack.pop()

                if not len(stack):
                    break

                # 이전과의 차이만큼 물 높이 처리
                distance = i - stack[-1] - 1
                waters = min(height[i], height[stack[-1]]) - height[top]

                volume += distance * waters

            stack.append(i)
        return volume
