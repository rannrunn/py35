import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1)
# 221 : 윗줄의 왼쪽에서 첫 번째
ax = fig.add_subplot(221, projection='3d')
ax.plot([1,2,3], [4,5,6], [1,2,3], 'b') # 파란색
# 222 : 윗줄의 왼쪽에서 두 번째
ax = fig.add_subplot(222, projection='3d')
ax.plot([1,2,3], [4,5,6], [1,3,5], 'g') # 초록색
# 223 : 아랫줄 왼쪽에서 첫 번째
ax = fig.add_subplot(223, projection='3d')
ax.plot([1,2,3], [4,5,6], [1,4,7], 'r') # 빨간색
# 224 : 아랫줄 왼쪽에서 두 번째
ax = fig.add_subplot(224, projection='3d')
ax.plot([1,2,3], [4,5,6], [1,5,9], 'k') # 검은색

plt.show()