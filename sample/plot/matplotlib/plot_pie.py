import numpy as np
import matplotlib.pyplot as plt


# 카테고리 별 값의 상대적인 비교를 해야 할 때는 pie 명령으로 파이 차트(pie chart)를 그릴 수 있다. 파이 차트를 그릴 때는 윈의 형태를 유지할 수 있도록 다음 명령을 실행해야 한다.
# plt.axis('equal')


labels = '개구리', '돼지', '개', '통나무'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()