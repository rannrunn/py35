import numpy as np
import matplotlib.pyplot as plt

# 바 차트와 유사하지만 폭(width)이 없는 스템 플롯(stem plot)도 있다. 주로 이산 확률 함수나 자기상관관계(auto-correlation)를 묘사할 때 사용된다.

x = np.linspace(0.1, 2 * np.pi, 10)
plt.stem(x, np.cos(x), '-.')
plt.show()


