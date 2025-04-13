# 완전 탐색
# Brute Force
# 무식하게 하는 것

# 1.
# 100만보다 작고2이상의 약수를 가지고 있으면, 틀린번호!

# TC = int(input())

# for _ in range(TC):
#     number = int(input())
#     for _ in range(2, 1000001):
#         if number % _ == 0:
#             print('NO')
#             break
#         if i == 1000000:
#             print('Yes')

# 앞으로 만나게 될 모든 문제를
# 완전탐색, 즉 모든 경우의 수를 넣어줄 것 이다.

# 모르는 문제를 만났을 때
# 어떤 문제든지, 경우의 수가 보인다면
# 우리는 시간과 메모리가 충분히 주어진다면
# 완전탐색으로 그 문제를 해결할 수 있다.

# 문제 2

# candy = int(input())

# answer = 0

# for A in range(0, candy+1):
#     for B in range(0, candy+1):
#         for C in range(0, candy+1):
#             if A+B+C == candy:
#                 if A >= B+2:
#                     if A != 0 and B != 0 and c != 0:
#                         if C % 2 == 0:
#                             answer += 1
# print(answer)

# 문제 3

# A, B, C, D, E, F = map(str, input().split())

# for x in range(-10000, 10000+1):
#     for y in range(-10000, 10000+1):
#         if A * x + B * y == C:
#             if D * x + E * y == F:
#                 print(x, y)
#                 break

# 문제 4
# n = int(input())
# hint = [list(map(str, input().split())) for _ in range(n)]
# answer = 0

# for a in range(1, 10):
#     for b in range(10):
#         for c in range(10):
#             if (a == b or b == c or c == a):
#                 continue

#             cnt = 0
#             for arr in hint:
#                 number = list(arr[0])
#                 ball = int(arr[1])
#                 strike = int(arr[2])

#                 ball_count = 0
#                 strike_count = 0

#                 if (a == int(number[0])):
#                     strike_count += 1
#                 if (b == int(number[1])):
#                     strike_count += 1
#                 if (c == int(number[2])):
#                     strike_count += 1

#                 if (a == int(number[1])) or (a == int(number[2])):
#                     ball_count += 1
#                 if (b == int(number[0])) or (b == int(number[2])):
#                     ball_count += 1
#                 if (c == int(number[0])) or (c == int(number[1])):
#                     ball_count += 1

#                 if (strike != strike_count):
#                     break
#                 if (ball != ball_count):
#                     break

#                 cnt += 1

#             if cnt == n:
#                 answer += 1

# print(answer)

# 문제 5

n = int(input())
arr = []
arr_y = []
arr_x = []
answer = [-1] * n

for _ in range(n):
    a, b = map(int, input().split())
    arr.append((a, b))
    arr_y.append(a)
    arr_x.append(b)

for y in arr_y:
    for x in arr_x:
        dist = []
        for ex, ey in arr:
            d = abs(ex - x) + abs(ey - y)
            dist.append(d)

        dist.sort()

        tmp = 0
        for i in range(len(dist)):
            d = dist[i]
            tmp += d
            if answer[i] == -1:
                answer[i] = tmp
            else:
                answer[i] = min(answer[i], tmp)

print(*answer)
