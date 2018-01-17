

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


objects = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
x_pos = np.arange(len(objects))

percent = 733268 / (733268 + 5656792)
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 88, 0, 0]
print(y)

plt.bar(x_pos, y, align='edge', alpha=1, width=0.97)
plt.xticks(x_pos, objects)
plt.ylim(0, 100)
plt.ylabel('Percent', fontsize=15)
plt.title('Press', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)

# bar에 text 추가 하기 위한 함수
def autolabel(bars):
    for ii, bar in enumerate(bars): # ii, bar의 의미를 모르겠음...
        height = '      ' + str(bars[ii])
        plt.text(x_pos[ii], height, '%s' % ('      ' + str(bars[ii])), ha='center', va='bottom', fontsize=12)

autolabel(y)

plt.show()