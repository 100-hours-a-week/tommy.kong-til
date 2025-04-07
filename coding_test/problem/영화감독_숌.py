# n = int(input())

# list_t = ['666']

# for i in range(1, 10):
#     list_t.append(str(i) + '666')

# for i in range(10, 100):
#     list_t.append(str(i) + '666')

# for i in range(10, 100):
#     list_t.append('666'+str(i))

# for i in range(100, 1000):
#     list_t.append(str(i) + '666')

# print(list_t[n-1])

# Solution
n = int(input())


def has_triple_six(num):
    return '666' in str(num)


count = 0
num = 666

while count < n:
    if has_triple_six(num):
        count += 1
    if count == n:
        break
    num += 1

print(num)
