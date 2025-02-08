import torch

# 행렬에 대한 인덱싱과 슬라이싱
A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]) # []가 하나의 행
print(A[0]) # 하나만 쓰면 행에 대한 인덱싱 (리스트 속 리스트 생각)
print(A[-1])
print(A[1:])
print(A[:])
print(A[0][2])
print(A[0,2]) # 2차원 행렬도 동일한데, 리스트와 달리 이런 것도 됨
B = [[1,2,3,4],[5,6,7,8]]
print(B[0][2])
# print(B[0,2]) # errol
print(A[1,:1])# 1 행, 전부
print(A[1,0:3:2])
print(A[:, 2]) # 행 전부, 2번째 열
print(A[:][2]) # A[0][2]와 다름. 처음부터 끝까지 인덱싱 why 행 전부 -> 여기서 두번째 행  


import numpy as np
array = np.arange(0,24).reshape((2,3,4))
print(array)
A = torch.tensor(array)
print(A)
print(A.shape)
print(A[0,1,2])

a = torch.tensor([1,2,3,4]) 
print(a.shape)
a = torch.tensor([[[1,2,3,4]]]) 
print(a.shape) # 대괄호가 하나 늘어나면 왼쪽에 shape값이 추가된다.
# 위와는 다르게 하나의 뭉탱이로 존재함. 1차원 행렬이 아님. 2차원 행렬임

print(A[[0,1,1,0],[0,1,2,1],[3,3,2,1]])



# boolean 인덱싱
a=[1,2,3,4,5,3,3]
print(a==3) # 여러개 값 들어있는 리스트랑 3 달랑 하나랑 같냐? 다르다!
A=torch.tensor([[1,2,3,4], [5,3,7,3]])
print(A>3) # 리스트와 달리 각 성분에 대해 비교해 줌
print(A[A>3]) # True, False가 담긴 행렬로 인덱싱 가능!!
A[A>3] = 100
print(A) # 그러면 이런 것도 가능하다! (3보다 큰 애들을 100으로 바꿔줘)
A=torch.tensor([[1,2], [3,4], [5,6], [7,8]])
B=torch.tensor([True, False, False, True])#행만 boo1로 인덱싱 하는 것도 가능
print(A[B,:]) # 0행, 3행 슬라이싱
b=torch.tensor([1,2,3,4])
print(b[[True,True,False,False]])# 참고로 그냥 리스트여도 됨
c=[1,2,3,4]
# c[[True True,False,Falsel]] # error
print()

# tensor로 인덱싱
a=torch.tensor([1,2,3,4,5])
A=a[2]
print(A)
A=a[torch.tensor(2)]
# torch.tensor를 안에다가?
print(A)
A=a[torch.tensor([2,3,4])]
print(A)
A=a[torch.tensor([[2,2,2], [3,3,3]])]
print(A)
# 인덱싱된 애들로 2행 3열짜리 행렬을 만든다
a=[1,2,3]
# a[[1,1,1,1,2,2,2]] #error!
a=torch.tensor([[1,2,3], [4,5,6]])
print(a[torch.tensor(0)])
A=a[torch.tensor([[0,1],[0,1]])]
print("A :", A)
print (A.shape)
# 예를들어, a[0l = tensor([1,2,3])과 같이 1차원 데이터이므로 한 차원이 뒤에 늘어나서 2,2, 3" 이 된다!
print(A) # segmentation 결과 그림 보여줄 때 사용!
# b=torch.tensor([[225,255,01, b=torch.tensor([[225,255,01, [O,255,01, [255,0,2551, [70,80,751, [0,0,4], [60,100,25511)
# import matplotlib.pyplot as pLt
# plt.imshow(bl torch.tensor([[0,1], [1,1]]) 1)


# 문제
a = torch.tensor([[1,2,3], [4,5,6]])

A = a[torch.tensor([[0,1],[1,1]])]
print(a)
print(A)

# 이를 리스트로 하려면?

print(a[[[0,0,0],[1,1,1]],[[0,1,2],[0,1,2]]])
# print(a[torch.tensor([0,1,1,1])])