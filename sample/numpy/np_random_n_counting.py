import numpy as np



# 시드 설정하기
# seed : pseudo random 상태 설정
np.random.seed(0)
# seed에 따라 0과 1사의 난수를 생성
print(np.random.rand(5))



# 기존 데이터의 순서 바꾸기
x = np.arange(10)
np.random.shuffle(x)
print(x)



# 기존 데이터에서 샘플링하기
# numpy.random.choice(a, size=None, replace=True, p=None)
# a : 배열이면 원래의 데이터, 정수이면 range(a) 명령으로 데이터 생성
# size : 정수. 샘플 숫자
# replace : 불리언. True이면 한번 선택한 데이터를 다시 선택 가능
# p : 배열. 각 데이터가 선택될 수 있는 확률

# shuffle 명령과 같다
print(np.random.choice(5, 5, replace=False))
# 3개만 선택
print(np.random.choice(5, 3, replace=False))
# 반복해서 10개 선택
print(np.random.choice(5, 10))
# 선택 확률을 다르게 해서 10개 선택
print(np.random.choice(5, 10, p=[0.1, 0, 0.3, 0.6, 0]))



# 난수 생성
# rand: 0부터 1사이의 균일 분포
# randn: 가우시안 표준 정규 분포
# randint: 균일 분포의 정수 난수

# 균일 분포로 10개 생성
print(np.random.rand(10))
# 3, 5 행렬로 난수 생성
print(np.random.rand(3, 5))

# 가우시안 분포로 10개 생성
print(np.random.randn(10))
# 3, 5 행렬로 난수 생성
print(np.random.randn(3, 5))

# 0에서 9까지의 숫자에서 랜덤으로 10개 생성
print(np.random.randint(10, size=10))
# 10에서 19까지의 숫자에서 랜덤으로 10개 생성
print(np.random.randint(10, 20, size=10))
# 10에서 19까지의 숫자에서 랜덤으로 3, 5 행렬 생성
print(np.random.randint(10, 20, size=(3,5)))



# 정수 데이터 카운팅
# unique 명령으로 중복값 제거
print(np.unique([11, 11, 2, 2, 34, 34]))
# 인덱스와 카운트 출력
a = np.array(['a', 'b', 'b', 'c', 'a'])
index, count = np.unique(a, return_counts=True)
print(index)
print(count)
# 주사위 처럼 특정 범위 안에 카운트를 세고 싶을 경우 bincount 안에 minlength 인수를 사용
print(np.bincount([1, 1, 2, 2, 2, 3], minlength=6))


