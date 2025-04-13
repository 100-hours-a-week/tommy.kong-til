# 완전 탐색 (Brute Force)

## 개념
완전 탐색 (Brute Force)은 가능한 모든 경우를 직접 탐색하여 정답을 찾는 방식입니다. 일반적으로 시간과 메모리가 충분할 때 사용할 수 있습니다. 경우의 수가 크지 않을 때 유용합니다.

---

## 문제 1: 비밀번호 확인 (소수 판별)
### 문제 설명
- 입력된 수가 적절한 비밀번호인지 확인하는 문제입니다.
- **모든 소인수가 1,000,000보다 크다면** 적절한 비밀번호입니다.
- 그렇지 않다면 `NO`를 출력합니다.

### 코드
```python
TC = int(input())

for _ in range(TC):
    number = int(input())
    for i in range(2, 1000001):
        if number % i == 0:
            print('NO')
            break
    else:
        print('YES')
```

### 풀이
- 2부터 1,000,000까지 나누어 보면서 나누어 떨어지는 경우 `NO`를 출력.
- 하나도 나누어 떨어지지 않으면 `YES` 출력.

---

## 문제 2: 사탕 나누기
### 문제 설명
- 친구 A, B, C에게 사탕을 나누어 주는 문제입니다.
- 조건:
  1. 남는 사탕이 없어야 함
  2. A는 B보다 2개 이상 많아야 함
  3. A, B, C는 사탕을 1개 이상 받아야 함
  4. C는 짝수 개의 사탕을 받아야 함

### 코드
```python
candy = int(input())
answer = 0

for A in range(0, candy+1):
    for B in range(0, candy+1):
        for C in range(0, candy+1):
            if A + B + C == candy:
                if A >= B + 2:
                    if A != 0 and B != 0 and C != 0:
                        if C % 2 == 0:
                            answer += 1

print(answer)
```

### 풀이
- 가능한 A, B, C의 모든 조합을 만들고 조건을 만족하는 경우를 카운트.

---

## 문제 3: 연립방정식 해결하기
### 문제 설명
- 주어진 연립방정식의 해를 찾아야 합니다.
- `A * x + B * y = C`
- `D * x + E * y = F`

### 코드
```python
A, B, C, D, E, F = map(int, input().split())

for x in range(-10000, 10001):
    for y in range(-10000, 10001):
        if A * x + B * y == C and D * x + E * y == F:
            print(x, y)
            break
```

### 풀이
- 가능한 `x, y` 조합을 탐색하며 두 식을 동시에 만족하는 해를 찾음.

---

## 문제 4: 숫자 야구
### 문제 설명
- 정답이 되는 세 자리 숫자를 찾는 문제.
- 각 숫자의 스트라이크와 볼을 확인하여 정답이 될 수 있는 모든 경우를 검토.

### 코드
```python
n = int(input())
hint = [list(map(str, input().split())) for _ in range(n)]
answer = 0

for a in range(1, 10):
    for b in range(10):
        for c in range(10):
            if a == b or b == c or c == a:
                continue

            cnt = 0
            for arr in hint:
                number = list(arr[0])
                ball = int(arr[1])
                strike = int(arr[2])

                ball_count = 0
                strike_count = 0

                if a == int(number[0]):
                    strike_count += 1
                if b == int(number[1]):
                    strike_count += 1
                if c == int(number[2]):
                    strike_count += 1

                if a == int(number[1]) or a == int(number[2]):
                    ball_count += 1
                if b == int(number[0]) or b == int(number[2]):
                    ball_count += 1
                if c == int(number[0]) or c == int(number[1]):
                    ball_count += 1

                if strike != strike_count or ball != ball_count:
                    break
                cnt += 1

            if cnt == n:
                answer += 1

print(answer)
```

### 풀이
- 1부터 9까지 서로 다른 세 자리 숫자를 만들고, 주어진 힌트와 비교하여 가능한 경우를 찾음.

---

## 문제 5: 최소 이동 거리 구하기
### 문제 설명
- 학생들이 모일 위치를 정하여 최소 이동 거리를 찾는 문제.
- N명의 학생들의 위치가 주어지며, 이동 거리 합이 최소가 되는 지점을 찾아야 함.

### 코드
```python
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
```

### 풀이
- 가능한 모든 위치에서 이동 거리 합을 계산하고 최소값을 찾음.

---

## 결론
- 완전 탐색은 가능한 모든 경우를 시도하여 답을 찾는 방식.
- 경우의 수가 많아질 경우 시간 복잡도를 고려하여 최적화 필요.
