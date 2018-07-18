import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


df = pd.read_csv('xclara.csv')
f1 = df['V1'].values
f2 = df['V2'].values
X = np.array(list(zip(f1, f2)))


plt.scatter(f1, f2, c='black', s=7)
plt.show()


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


k = 3
C_x = np.random.randint(0, np.max(X) - 20, size=k)
C_y = np.random.randint(0, np.max(X) - 20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)


C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
error = dist(C, C_old, None)


# 핵심 알고리즘
while error != 0:
    # 포인트 별 가장 가까운 클러스터에 대한 인덱스 저장
    for i in range(len(X)):
        distances = dist(X[i], C)
        clusters[i] = np.argmin(distances)
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)


colors = ['r', 'g', 'b', 'y']
# 점 표시
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], c=colors[i], s=7)
ax.scatter(C[:, 0], C[:, 1], c='#050505', s=200, marker='*')

plt.show()

