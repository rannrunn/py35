import numpy as np
import matplotlib.pyplot as plt

# x 데이터가 카테고리 값인 경우에는 bar 명령과 barh 명령으로 바 차트(bar chart) 시각화를 할 수 있다. 가로 방향으로 바 차트를 그리려면 barh 명령을 사용한다.
# 바 차트 작성시 주의점은 첫번째 인수인 left 가 x축에서 바(bar)의 왼쪽 변의 위치를 나타낸다는 점이다.
y = [2, 3, 1]
x = np.arange(len(y))
xlabel = ['가', '나', '다']
plt.bar(x, y)
plt.xticks(x, xlabel)
plt.show()


d

# xerr 인수나 yerr 인수를 지정하면 에러 바(error bar)를 추가할 수 있다.
#
# 다음 코드에서 alpha는 투명도를 지정한다. 0이면 완전 투명, 1이면 완전 불투명이다.

np.random.seed(0)

people = ['몽룡', '춘향', '방자', '향단']
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.barh(y_pos, performance, xerr=error, alpha=0.4)
plt.yticks(y_pos, people)
plt.xlabel('x 라벨')
plt.show()