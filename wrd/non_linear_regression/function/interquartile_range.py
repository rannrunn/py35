num_friends = [100,15,10,10,9,4,3,3,2,1]


def quartile(x,list,p):
    p_index = int(p * len(x))
    return list[p_index]

def interquartile_range(x):
    sorted_list = sorted(x)
    return quartile(x,sorted_list,0.75) - quartile(x,sorted_list,0.25)

print(interquartile_range(num_friends))