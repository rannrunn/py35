import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

objects = (0, 100, 1000, 5000, 10000, 40000, 82000)
x_pos = np.arange(len(objects))

x = [6359292, 31, 27, 0, 0, 0, 0]
y = [(item / sum(x)) * 100 for item in x]
print(y)
y = [99.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
print(y)
# y_label = [99.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]

plt.bar(x_pos, y, align='center', alpha=1, width=0.9) # 글자를 가운데 위치하기 위해 align을 center로 수정
plt.xticks(x_pos, objects)
plt.ylim(0, 110) # 글자가 안보여서 110으로 늘림
plt.ylabel('Percent', fontsize=15)
plt.title('Ambient', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)

# bar에 text 추가 하기 위한 함수
def autolabel(bars):
    for ii, bar in enumerate(bars): # ii, bar의 의미를 모르겠음...
        height = bars[ii]
        plt.text(x_pos[ii], height, '%s' % (bars[ii]), ha='center', va='bottom', fontsize=12)

autolabel(y)
plt.show()
