

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

all = [6359315, 6, 5, 1, 6, 4, 0, 0, 1, 1, 2, 7, 2]
print(sum(all))
objects = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
x_pos = np.arange(len(objects))
cnt = [20, 6, 5, 1, 6, 4, 0, 0, 1, 1, 2, 7, 2]
print(sum(cnt))
y = [(item / sum(cnt)) * 100 for item in cnt]
print(y)

plt.bar(x_pos, cnt, align='edge', alpha=1, width=0.97)
plt.xticks(x_pos, objects)
plt.ylim(0, 20)
plt.ylabel('횟  수', fontsize=15)
plt.title('UV', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)


plt.show()