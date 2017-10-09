from __future__ import division

num_friends = [1200,15,10,10,9,4,3,3,2]

def mean(x):
    return sum(x) / len(x)

avgOfFriends = mean(num_friends)
print (avgOfFriends)

def median(v):
    n = len(v)
    sorted_v = sorted(v) # 정렬해준 뒤에
    midpoint = n // 2 # // 로 나누면 int형이 됨. /로 나누면 float

    if n % 2 == 1:
        return sorted_v[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint + 1
        return (sorted_v[lo] + sorted_v[hi] ) / 2

medianOfFriends = median(num_friends)
print (medianOfFriends)

