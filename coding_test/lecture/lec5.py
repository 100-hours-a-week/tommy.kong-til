# 재귀 함수 -> for 반복문을 재귀함수로 바꿔줄 수 있다.

import sys
N, M = map(int, input().split())
arr = []


def recur(number):
    if number == M:
        print(*arr)
        return  # return을 통해 끝내준다.
    for i in range(1, N+1):
        arr.append(i)
        recur(number+1)
        arr.pop()


recur(0)

N, M = map(int, input().split())
arr = []


def recur(number):
    if number == M:
        print(*arr)
        return  # return을 통해 끝내준다.
    for i in range(1, N+1):
        if i in arr:
            continue
        arr.append(i)
        recur(number+1)
        arr.pop()


recur(0)

N, M = map(int, input().split())
arr = []


def recur(number):
    if number == M:
        print(*sorted(arr))
        return  # return을 통해 끝내준다.
    for i in range(1, N+1):
        if i in arr:
            continue
        arr.append(i)
        recur(number+1)
        arr.pop()


recur(0)

N, M = map(int, input().split())
arr = []


def recur(start, number):
    if number == M:
        print(*arr)
        return
    for i in range(start, N + 1):  # i부터 시작하도록 변경 (중복 방지)
        arr.append(i)
        recur(i + 1, number + 1)  # 다음 재귀 호출에 i+1을 전달하여 중복 제거
        arr.pop()


recur(1, 0)

N, M = map(int, input().split())
arr = []


def recur(start, number):
    if number == M:
        print(*arr)
        return
    for i in range(N, start - 1, -1):  # N부터 시작해 점점 작아지도록 변경
        arr.append(i)
        recur(i - 1, number + 1)  # 다음 숫자는 i-1부터 선택하도록 변경
        arr.pop()


recur(N, 0)


sys.setrecursionlimit(9999999)


def recur(hint_idx, number):

    if hint_idx == 4:
        answer += 1
        recur(0, number+1)
        return

    if number == 1000:
        return

    if  # 만약에 힌트에 통과했다면

    recur(hint_idx+1, number)
    if  # 만약에 힌트에 통과하지 않았다면
    recur(0, number+1)


n = int(input())

hint = [list(map(int, input().split())) for _ in range(n)]

answer = 0

recur(0, 100)
