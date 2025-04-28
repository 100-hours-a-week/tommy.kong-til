#
# @lc app=leetcode id=234 lang=python3
#
# [234] Palindrome Linked List
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        l = []
        
        node = head
        
        while node is not None:
            l.append(node.val)
            node = node.next
            
        for i in range(len(l)):
            if l[i] == l.pop():
                pass
            else:
                return False
            return True
                
        
# @lc code=end

