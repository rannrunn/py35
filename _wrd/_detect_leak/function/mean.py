from __future__ import division

num_friends = [100,40,30,30,30,30,30,30,30,30,54,54,54,54,54,54,54,54,25,3,100,100,100,3,3]

def mean(x):
    return sum(x) / len(x)

avgOfFriends = mean(num_friends)
print (avgOfFriends)

