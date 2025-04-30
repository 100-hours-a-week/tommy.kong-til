class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None:
            return head

        else:
            cur, cur.next = head, head.next

            while cur and cur.next:
                cur.val, cur.next.val = cur.next.val, cur.val
                cur = cur.next.next

            return head

# 반복 구조로 스왑


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        root = prev = ListNode(None)
        prev.next = head
        while head and head.next:
            b = head.next
            head.next = b.next
            b.next = head

            prev.next = b

            head = head.next
            prev = prev.next

        return root.next

# 재귀로 풀기


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head and head.next:
            p = head.next
            head.next = self.swapPairs(p.next)  # 4->3->null 을 반환한다. 이것이 head
            p.next = head  # p =2 인데 2 노드는 이제 1을 가리킴. 1은 이미 4->3->null을 가리키고 있기 때문에 2->1->4->3->null의 구조를 가짐
            return p
        return head

# 리스트 1->2->3->4의 변환:
# 첫 호출: head=1, p=2
# 재귀 호출: swapPairs(3)
# 여기서 head=3, p=4
# 재귀 호출: swapPairs(null) -> 기본 케이스, 반환 null
# head=3의 next는 null이 됨
# p=4의 next는 head=3이 됨
# 반환: p=4 (즉 4->3->null)
# head=1의 next는 4->3->null이 됨
# p=2의 next는 head=1이 됨
# 반환: p=2 (즉 2->1->4->3->null)
# 결과: 2->1->4->3
