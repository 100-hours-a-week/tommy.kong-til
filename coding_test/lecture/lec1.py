# 강의 1

# 입력을 받고 (디테일!)
# default input is string type

# 배열

list = [0, 0, 0]
list = ['he', 'hi']

# case 1 : 단순히 정수 일때,
number = int(input())

# case 2: 수열
first, second, third = map(int, input().split())
list1 = list(map(int, input().split()))

# case 3 : 단순히 문장 일때,
string = input()

# case 4 : 문자열
first, second, third = map(str, input().split())
list2 = list(map(str, input().split()))

# 계산을 하고


# 출력을 한다.

print(*list1)
print(*list2)
# 배열 속에 잇는 내용을 그대로 출력 한다.


# 반복문
for _ in range(200):
    print('hi')

while _ < 10:
    print(_)
    _ = _ + 1

# 조건문
name = 't'
if name == 't':
    print(f"your name is {name}")
else:
    print(f'your name is not {name}')
