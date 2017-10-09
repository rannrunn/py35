import collections
from matplotlib import pyplot as plt

num_firends = [100,40,30,30,30,30,30,30,30,30,54,54,54,54,54,54,54,54,25,3,100,100,100,3,3]
friend_counts = collections.Counter(num_firends)
print('friends:', friend_counts)

xs = range(101)
ys = [friend_counts[x] for x in xs]

#print(xs)

plt.bar(xs,ys)
plt.axis([0,101,0,25])
plt.xlabel(" # of friends")
plt.ylabel(" # of people")
plt.show()
