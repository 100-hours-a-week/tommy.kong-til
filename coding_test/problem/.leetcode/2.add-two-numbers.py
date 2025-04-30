#
# @lc app=leetcode id=2 lang=python3
#
# [2] Add Two Numbers
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = None
        while not head:
            head, l1 = l1, l1.next
            if head.val >= l1.val:
                l1.next, head = head, l1

        while not head:
            head, l2 = l2, l2.next
            if head.val >= l2.val:
                l2.next, head = head, l2

        while l1:
            result = l1+l2
            l1.next
            l2.next

        return result


# @lc code=end
