import matplotlib.pyplot as plt
import numpy as np

n = 5
data = np.array(range(n)) + np.random.rand(n)
print(data)
fig, ax = plt.subplots(3, figsize=(6, 9))

bar_locations = np.arange(n)
print(bar_locations)
ax[0].bar(bar_locations, data)
ax[1].bar(bar_locations, data, align='edge')
ax[2].bar(bar_locations, data, align='center')

plt.show()