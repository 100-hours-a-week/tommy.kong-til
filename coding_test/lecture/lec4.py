# 3강 누적합 (기억)

# 컴퓨터는 이전에 저장하고 있던 값을 이용해서 문제를 풀어나간다.
# 컴퓨터에게 기억하는 방법을 알려주는 것


# 완탐 For > 재귀함수(백트래킹)
# 최적화 정수론 > 재귀함수/백트래킹의 경우의 수
# 기억 누적합 > 탑다운 DP, 바텀업DP 메모라이제이션

# 문제 1

# n, r = map(int, input().split())
# arr = list(map(int, input().split()))
# i = []

# for _ in range(n):
#     if _ == 0:
#         i.append(arr[_])
#     else:
#         i.append(i[_-1] + arr[_])

# sum_l = []
# for _ in range(n-r):
#     sum_l.append(i[_+r] - i[_])

# print(max(sum_l))

# a, b = map(int,input().split())

# array = list(map(int,input().split()))

# prefix = [0 for _ in range(a+1)]

# for i in range(0,a):10
#     prefix[i+1] = prefix[i] + array[i]

# answer = []
# for k in range(b,a+1):
#     answer.append(prefix[k] - prefix[k-b])

# print(max(answer))

# 다이내믹 프로그래밍(메모이제이션)

# 문제 2
# n = int(input())
# arr = list(map(int, input().split()))
# prefix = [0 for _ in range(n+1)]

# for i in range(n):
#     prefix[i+1] = max(prefix[i] + arr[i], arr[i])

# print(max(prefix))


# 문제 3
n = int(input())
arr = [list(map(int, input().split())) for _ in range(n)]
prefix = [0 for _ in range(n+1)]

for i in range(n):
    for j in range(n):

        # 문제 4
        # arr = [list(map(int, input().split())) for _ in range(4)]
        # x1, y1, x2, y2 = map(int, input().split())
        # prefix = [[0 for _ in range(5)] for _ in range(5)]

        # for i in range(4):
        #     for j in range(4):
        #         prefix[i+1][j+1] = prefix[i][j+1] + \
        #             prefix[i+1][j] - prefix[i][j] + arr[i][j]

        # print(prefix[x2][y2] - prefix[x1-1][y2] -
        #       prefix[x2][y1-1] + prefix[x1-1][y1-1])

        # 문제 5
        # import sys
        # input = sys.stdin.readline

        # n, h = map(int, input().split())

        # line = [0 for _ in range(h)]

        # for t in range(n):
        #     height = int(input())
        #     if t % 2 == 0:
        #         line[0] += 1
        #         line[height] -= 1

        #     if t % 2 == 1:
        #         line[h-height] += 1

        # print(line)

        # prefix = [0 for _ in range(h+1)]

        # for i in range(h):
        #     prefix[i+1] = prefix[i] + line[i]

        # prefix = prefix[1:]

        # print(min(prefix), prefix.count(min(prefix)))
