
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(node: ListNode, prev: ListNode = None):
            if not node:
                return prev
            next, node.next = node.next, prev
            return reverse(next, node)
        return reverse(head)


# 입력 연결 리스트: 1 -> 2 -> 3 -> 4 -> 5 -> None

# 1. 첫 번째 호출: reverse(1, None)
    # node = 1, prev = None
    # next = 2 (노드 1의 다음 노드)
    # node.next = prev: 1의 next를 None으로 변경 (1 -> None)
    # 재귀 호출: reverse(2, 1)
# 2. 두 번째 호출: reverse(2, 1)
    # node = 2, prev = 1
    # next = 3 (노드 2의 다음 노드)
    # node.next = prev: 2의 next를 1로 변경 (2 -> 1 -> None)
    # 재귀 호출: reverse(3, 2)
# 3. 세 번째 호출: reverse(3, 2)
    # node = 3, prev = 2
    # next = 4 (노드 3의 다음 노드)
    # node.next = prev: 3의 next를 2로 변경 (3 -> 2 -> 1 -> None)
    # 재귀 호출: reverse(4, 3)
# 4. 네 번째 호출: reverse(4, 3)
    # node = 4, prev = 3
    # next = 5 (노드 4의 다음 노드)
    # node.next = prev: 4의 next를 3으로 변경 (4 -> 3 -> 2 -> 1 -> None)
    # 재귀 호출: reverse(5, 4)
# 5. 다섯 번째 호출: reverse(5, 4)
    # node = 5, prev = 4
    # next = None (노드 5의 다음 노드)
    # node.next = prev: 5의 next를 4로 변경 (5 -> 4 -> 3 -> 2 -> 1 -> None)
    # 재귀 호출: reverse(None, 5)
# 6. 여섯 번째 호출: reverse(None, 5)
    # node = None, prev = 5
    # 기저 조건 충족: not node가 True
    # return prev: 노드 5를 반환 (새로운 헤드)
# 최종 결과
    # 원래 리스트: 1 -> 2 -> 3 -> 4 -> 5 -> None
    # 역순 변환 후: 5 -> 4 -> 3 -> 2 -> 1 -> None


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node, prev = head, None

        while node:
            next, node.next = node.next, prev
            prev, node = node, next

        return prev

# 예시를 통한 작동 방식
# 입력 연결 리스트: 1 -> 2 -> 3 -> 4 -> 5 -> None

# 1. 초기 상태
    # node = 1 (head)
    # prev = None
# 2. 첫 번째 반복
    # next = 2 (노드 1의 다음 노드)
    # node.next = prev: 1의 next를 None으로 변경 (1 -> None)
    # prev = 1, node = 2 (포인터 이동)
    # 현재 상태: None <- 1   2 -> 3 -> 4 -> 5 -> None
# 3. 두 번째 반복
    # next = 3 (노드 2의 다음 노드)
    # node.next = prev: 2의 next를 1로 변경 (2 -> 1 -> None)
    # prev = 2, node = 3 (포인터 이동)
    # 현재 상태: None <- 1 <- 2   3 -> 4 -> 5 -> None
# 4. 세 번째 반복
    # next = 4 (노드 3의 다음 노드)
    # node.next = prev: 3의 next를 2로 변경 (3 -> 2 -> 1 -> None)
    # prev = 3, node = 4 (포인터 이동)
    # 현재 상태: None <- 1 <- 2 <- 3   4 -> 5 -> None
# 5. 네 번째 반복
    # next = 5 (노드 4의 다음 노드)
    # node.next = prev: 4의 next를 3으로 변경 (4 -> 3 -> 2 -> 1 -> None)
    # prev = 4, node = 5 (포인터 이동)
    # 현재 상태: None <- 1 <- 2 <- 3 <- 4   5 -> None
# 6. 다섯 번째 반복
    # next = None (노드 5의 다음 노드)
    # node.next = prev: 5의 next를 4로 변경 (5 -> 4 -> 3 -> 2 -> 1 -> None)
    # prev = 5, node = None (포인터 이동)
    # 현재 상태: None <- 1 <- 2 <- 3 <- 4 <- 5
# 7. 반복문 종료
    # node = None이므로 반복문 종료
    # return prev: 노드 5를 반환 (새로운 헤드)
# 최종 결과
    # 원래 리스트: 1 -> 2 -> 3 -> 4 -> 5 -> None
    # 역순 변환 후: 5 -> 4 -> 3 -> 2 -> 1 -> None
# 핵심 사항
    # 3개의 포인터 사용: prev, node, next를 사용하여 연결을 안전하게 변경
    # 동시 할당: Python의 다중 할당 문법을 활용하여 간결한 코드 작성
    # next, node.next = node.next, prev: 다음 노드 저장 및 현재 노드 방향 변경
    # prev, node = node, next: 포인터 이동
