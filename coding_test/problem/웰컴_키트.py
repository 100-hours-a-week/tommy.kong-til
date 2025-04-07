N = int(input())
T_list = list(map(int, input().split()))
G = list(map(int, input().split()))
T, P = G[0], G[1]

T_C = 0

for i in range(len(T_list)):
    if T_list[i] % T == T_list[i] or T_list[i] == T:
        T_C += 1
    elif T_list[i] % T == 0:
        T_C += T_list[i] // T
    else:
        T_C += (T_list[i] // T) + 1

P_C = N // P
P_S = N % P

print(T_C)
print(P_C, P_S)
