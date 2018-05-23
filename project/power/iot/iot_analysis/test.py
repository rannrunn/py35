import scipy.signal
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
sp.random.seed(0)
N = 2**15
e = sp.stats.bernoulli.rvs(0.5, size=N) * 2 - 1

f1, P1 = sp.signal.welch(e)
plt.semilogy(f1, P1);

# 비교를 위한 단일 주파수 신호 (mono tone)
fs = 10e3; N = 1e5; amp = 2*np.sqrt(2); freq = 1000; noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
s = amp*np.sin(2*np.pi*freq*time)

f2, P2 = sp.signal.welch(s)
plt.semilogy(f2, P2);
plt.xlim([0.01, 0.49])
plt.ylim([0.5e-1, 1e3])
plt.show()