import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
digits = load_digits()
X = digits.images[0]

# 화상(image) 데이터처럼 행과 열을 가진 행렬 형태의 2차원 데이터는 imshow 명령을 써서 2차원 자료의 크기를 색깔로 표시하는 것이다.


plt.imshow(X, interpolation='nearest', cmap=plt.cm.bone_r)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.subplots_adjust(left=0.35, right=0.65, bottom=0.35, top=0.65)
plt.show()



# 데이터 수치를 색으로 바꾸는 함수는 칼라맵(color map)이라고 한다. 칼라맵은 cmap 인수로 지정한다. 사용할 수 있는 칼라맵은 plt.cm의 속성으로 포함되어 있다. 아래에 일부 칼라맵을 표시하였다. 칼라맵은 문자열로 지정해도 된다. 칼라맵에 대한 자세한 내용은 다음 웹사이트를 참조한다.

dir(plt.cm)[:10]

fig, axes = plt.subplots(1, 4, figsize=(12, 3),
                         subplot_kw={'xticks': [], 'yticks': []})
axes[0].imshow(X, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].imshow(X, interpolation='nearest', cmap=plt.cm.Blues_r)
axes[2].imshow(X, interpolation='nearest', cmap='BrBG')
axes[3].imshow(X, interpolation='nearest', cmap='BrBG_r')
plt.show()


# imshow 명령은 자료의 시각화를 돕기위해 다양한 2차원 인터폴레이션을 지원한다.

methods = [
    None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
]
fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(X, cmap=plt.cm.bone_r, interpolation=interp_method)
    ax.set_title(interp_method)
plt.show()