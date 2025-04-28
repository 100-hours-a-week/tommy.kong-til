#
# @lc app=leetcode id=21 lang=python3
#
# [21] Merge Two Sorted Lists
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        l1, l2 = list1, list2
        # l1,l2의 값을 비교해 작은 값이 왼쪽에 오게 하고 next는 그다음 값이 엮이도록 재귀호출
        if (not l1) or (l2 and l1.val > l2.val):
            # 1. 비교연산
            # 2. not l1 -> 바로 l1,l2 스왑
            # 3. or (l2  and l1.val > l2.val) -> l1이면 해당 조건문을 확인해서 l1,l2 스왑
            #
            l1, l2 = l2, l1
        if l1:
            l1.next = self.mergeTwoLists(l1.next, l2)
            # l1의 next를 계쏙 호출하면서 계속 스왑될 수 있게
        return l1


# 재귀 동작 예시
# 입력이 list1 = [1,2,4], list2 = [1,3,4]인 경우:
    # 첫 호출: l1=[1,2,4], l2=[1,3,4] 값이 같으므로 스왑 없음
    # 재귀 호출: mergeTwoLists([2,4], [1,3,4])
# 두 번째 호출: l1=[2,4], l2=[1,3,4]
    # l1.val > l2.val이므로 스왑: l1=[1,3,4], l2=[2,4]
    # 재귀 호출: mergeTwoLists([3,4], [2,4])
# 세 번째 호출: l1=[3,4], l2=[2,4]
    # l1.val > l2.val이므로 스왑: l1=[2,4], l2=[3,4]
    # 재귀 호출: mergeTwoLists([4], [3,4])
# 이후 과정 반복

# @lc code=end
