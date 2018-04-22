import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


objects = ('고창','대구', '광주', 'Test')
x_pos = np.arange(len(objects))

x = [6359292, 31, 27, 0, 0, 0, 0]
print(sum(x))
y = [(item / sum(x)) * 100 for item in x]
print(y)
y = [99.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
print(y)

plt.bar(x_pos, y, align='edge', alpha=1, width=0.97)
plt.xticks(x_pos, objects)
plt.ylim(0, 100)
plt.ylabel('Percent', fontsize=15)
plt.title('Ambient', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)

plt.show()