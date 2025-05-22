# N = int(input())
# interview = [list(map(int, input().split())) for _ in range(N)]

# # 모든 상담을 확인 하기
# total_income = []
# income = (
#     interview[i][1]
#     + interview[i + interview[i][0]][1]
#     + interview[i + interview[i + interview[i][0]][0]][1]
# )  # .. 계속 더해주는 느낌으로
# 자체를 하나의 함수로 만들어주면 되겠다.


# def income(i):
#     return interview[i][1] + interview[i + interview[i][0]][1]

# income_list = []
# for i in range(N):
#     total_income = 0
#     k = i
#     while k < N:
#         if k == N - 1 and interview[N - 1][0] != 1:
#             break
#         elif k == N - 1 and interview[N - 1][0] == 1:
#             total_income += interview[N - 1][1]
#             break
#         elif k + interview[k][0] > N:
#             break
#         else:
#             total_income += interview[k][1]
#             k += interview[k][0]
#     income_list[i] = total_income

# print(max(income_list.values()))

# N = int(input())
# interview = [list(map(int, input().split())) for _ in range(N)]

# income_list = []
# for i in range(N):
#     total_income = 0
#     k = i
#     while k < N:
#         if k + interview[k][0] > N:  # 상담이 기간을 초과하는 경우
#             # 이 조건은 마지막 날을 포함한 모든 경우를 처리합니다.
#             # 만약 상담이 마지막 날에 완료될 수 있다면(k + interview[k][0] <= N), 해당 상담의 수익이 total_income에 추가됩니다.
#             # 마지막 날의 상담 시간이 1이라면, k + interview[k][0]은 정확히 N이 되므로 조건을 만족하고 수익에 포함됩니다.
#             break
#         total_income += interview[k][1]
#         k += interview[k][0]
#     income_list.append(total_income)  # 리스트에 값 추가

# print(max(income_list))  # 최대 수익 출력


# N = int(input())
# interview = [list(map(int, input().split())) for _ in range(N)]

# # DP 테이블 초기화
# dp = [0] * (N + 1)

# # 뒤에서부터 최대 수익 계산
# for i in range(N - 1, -1, -1):
#     time, pay = interview[i]
#     if i + time <= N:  # 상담이 가능한 경우
#         dp[i] = max(dp[i + 1], pay + dp[i + time])
#     else:  # 상담이 불가능한 경우
#         dp[i] = dp[i + 1]

# # 결과 출력
# print(dp[0])

def max_income(day, total_income):
    if day >= N:  # 상담 기간을 초과한 경우
        return total_income

    # 상담을 선택하지 않는 경우
    skip = max_income(day + 1, total_income)

    # 상담을 선택하는 경우
    if day + interview[day][0] <= N:
        take = max_income(day + interview[day][0], total_income + interview[day][1])
    else:
        take = 0

    return max(skip, take)


N = int(input())
interview = [list(map(int, input().split())) for _ in range(N)]

print(max_income(0, 0))