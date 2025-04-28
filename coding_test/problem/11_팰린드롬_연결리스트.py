# Sol1 리스트 변환 

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        l = []
        
        node = head
        
        while node is not None:
            l.append(node.val)
            node = node.next
        
        return l == l[::-1]

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        l = []
        
        node = head
        
        while node is not None:
            l.append(node.val)
            node = node.next
            
        left, right = 0, len(l) - 1
        while left < right:
            if l[left] != l[right]:
                return False
            left += 1
            right -= 1
            
        return True
    
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        q: List = []
        
        node = head
        
        while node is not None:
            l.append(node.val)
            node = node.next
            
        while len(q) > 1:
            if q.pop(0) != q.pop():
                return False
        
        return True
        


# Sol2 데크를 이용한 최적화 

# 파이썬의 데크는 이중 연결 리스트 구조로 양쪽 방향 모두 추출하는 데 시간복잡도 O(1)에서 실행된다. 
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        q: Deque = collections.deque()

        if not head:
            return True
        
        node = head
        while node is not None:
            q.append(node.val)
            node = node.next
        
        while len(q) > 1:
            if q.popleft() != q.pop():
                return False
        
        return True 
# 데크를 명시적으로 선언만으로 상당한 속도 개선이 가능 

# Sol3 런너를 이용한 풀이 
# 연결 리스트를 순회할 때 2개의 포인터를 동시에 사용하는 기법 
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        rev = None
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev, slow.next
        if fast: # 홀수있대는 slow런너가 한 칸 더 앞으로 이동해야함.
            # 이는 fast가 아직 None이 아니라는 경우로 간주할 수 있으며,따라서 이 경우 다음과 같이slow를 한 칸 더 이동해 마무리한다
            slow = slow.next
        
        while rev and rev.val == slow.val:
            slow, rev = slow.next, rev.next
        return not rev

# 다중 할당    
rev, rev.next, slow = slow, rev, slow.next
# 서로 다르다. 
rev,rev.next = slow,rev
slow = slow.next
# 왜랴하면 rev와 slow는 서로 같은 값을 참조하게 됨.
# = 연산자를 이용해 할당을 진행하게 되면 값을 할당하는 것이 아니라 이 불변 객체에 대한 참조를 함

id(5)
a = 5
id(a)
b = 5
id(b)

# 5라는 숫자에 대해 숫자5와 변수a,b모두 동일한ID를 갖는다.
# 즉 5라는 값은 메모리 상에 단 하나만 존재하며,a,b 두 변수는 각각 이 값을 가리키는 참조라는 의미다. 
# 만약 5가 6으로 변경된다면, a,b두 변수도 값이 6으로 변경될 것이다.
# 그러나 이런 일은 일어나지 않는다.앞서 설명한 것처럼 숫자는 불변 객체이기 때문이다. 
# 반면 숫자가 아니라 리스트와 같은 자료형이라면, 내부의 값은 얼 마든지 변할 수 있다. 
# 이 경우 이 리스트를 참조하는 모든 변수의 값도 따라서 함께 바뀌게 된다. 
# 이번에는 문제 풀이에서처럼 rev = 1, sLow = 2->3이라고 가정해보자. 
# 여기서 SLow는 연결 리스 트이며, slow.next는 3이라는 의미다.

# 이번에는 문제 풀이에서처럼 rev = 1, sLow = 2->3이라고 가정해보자. 
# 여기서 SLow는 연결 리스 트이며, slow.next는 3이라는 의미다.
rev, rev.next, slow = slow, rev, slow.next

# 이 경우 rev = 2-23, rev.next = 1, SLow = 30| 되고, rev.next = 10므로 최종적으로 rev =2-21,sLow=30| 된다. 
# 다중 할당을 하게 되면 이 같은 작업이 동시에 일어나기 때문에, 이 모 든 작업은 중간 과정 없이 한 번의 트랜잭션으로 끝나게 된다. 
# 그러나 앞서 살펴본 두 줄 분기 코드 인 다음과 같이 나눠서 처리하는 경우를 생각해보자.

rev, rev.next = slow, rev slow = slow.next
# 첫 줄을 실행한 결과, rev = 2-23, rev.next = 1따라서 rev = 2->101 되는데 여기서 중요한 점은 rev = slow라는 점이다. 
# 즉 동일한 참조가 되었으며 rev = 2->1이 되었기 때문에 slow = 2->1도 함께 되어 버린다. 따라서 이후에 SLow = Slow.next의 결과는 SLow = 10 된다. 
# 결국. 최종 결과는 rev=2-21,slow=1로, 앞서 다중 할당으로 한 번에 처리한 것과 다른 결과가 된 다.따라서 앞서 풀이의 경우,반드시 한 줄의 다중 할당으로 한 번에 처리해야 문제를 제대로 풀이 할 수 있다.
# 이제 왜 이렇게 다중 할당을 하는지, 나누지 않고 한 번에 처리해야 하는지 어느 정도 이해가 될 것 이다. 
# 이런 부분은 파이썬의 중요한 특징이므로 충분한 숙지가 필요하다.
#  특히 온라인 코딩 테스트 시 이런 문제 때문에 헤매게 된다면,오히려 자바나C++로 풀이할 때보다 시간을 더 허비할 수 있 으니 주의가 필요하다. 
# 파이썬의 강력함을 십분 활용하기 위해서는 이러한 파이썬의 특징을 잘 익 혀두기 바란다.