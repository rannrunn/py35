import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


t = np.arange(10)
np.random.seed(99)
y1 = np.insert(np.cumsum(stats.norm.rvs(size=9)), 0, 0)
y2 = np.insert(np.cumsum(stats.norm.rvs(size=9)), 0, 0)
y3 = np.insert(np.cumsum(stats.norm.rvs(size=9)), 0, 0)
y4 = np.insert(np.cumsum(stats.norm.rvs(size=9)), 0, 0)

ax1 = plt.subplot(4, 1, 1)
ax1.plot(t, y1, '-o')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_ylim(-5, 5)
ax1.set_zorder(2)
ax1.set_ylabel("sample 1").set_rotation(0)
ax1.yaxis.set_tick_params(pad=30)

ax2 = plt.subplot(4, 1, 2)
ax2.plot(t, y2, '-o')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_ylim(-5, 5)
ax2.set_zorder(1)
ax2.set_ylabel("sample 2").set_rotation(0)
ax2.yaxis.set_tick_params(pad=30)

ax3 = plt.subplot(4, 1, 3)
ax3.plot(t, y3, '-o')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_ylim(-5, 5)
ax3.set_zorder(1)
ax3.set_ylabel("sample 3").set_rotation(0)
ax3.yaxis.set_tick_params(pad=30)

ax4 = plt.subplot(4, 1, 4)
ax4.plot(t, y4, '-o')
ax4.set_yticklabels([])
ax4.set_ylim(-5, 5)
ax4.set_zorder(1)
ax4.set_ylabel("sample 4").set_rotation(0)
ax4.yaxis.set_tick_params(pad=30)
ax4.annotate('$Y_6$', xy=(6 - 0.05, -11), xycoords='data', annotation_clip=False)

from matplotlib.patches import ConnectionPatch
con = ConnectionPatch(xyA=(6,5), xyB=(6,-5), ls="--", lw=2, color="gray",
                      coordsA="data", coordsB="data", axesA=ax1, axesB=ax4)
ax1.add_artist(con);


plt.show()