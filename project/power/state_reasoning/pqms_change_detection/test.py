import numpy as np
import matplotlib.pyplot as plt

from changepy.costs import normal_mean





import numpy as np

def find_min(arr, val=0.0):
    """ Finds the minimum value and index

    Args:
        arr (np.array)
        val (float, optional): value to add
    Returns:
        (float, int): minimum value and index
    """
    return min(arr) + val, np.argmin(arr)



def pelt(cost, length, pen=None):
    """ PELT algorithm to compute changepoints in time series

    Ported from:
        https://github.com/STOR-i/Changepoints.jl
        https://github.com/rkillick/changepoint/
    Reference:
        Killick R, Fearnhead P, Eckley IA (2012) Optimal detection
            of changepoints with a linear computational cost, JASA
            107(500), 1590-1598

    Args:
        cost (function): cost function, with the following signature,
            (int, int) -> float
            where the parameters are the start index, and the second
            the last index of the segment to compute the cost.
        length (int): Data size
        pen (float, optional): defaults to log(n)
    Returns:
        (:obj:`list` of int): List with the indexes of changepoints
    """
    if pen is None:
        pen = np.log(length) # 5.703782474656201

    F = np.zeros(length + 1)
    R = np.array([0], dtype=np.int) # array([0])
    candidates = np.zeros(length + 1, dtype=np.int) #

    F[0] = -pen

    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = np.zeros(len(cpt_cands))
        for i in range(0, len(cpt_cands)):
            seg_costs[i] = cost(cpt_cands[i], tstar)

        F_cost = F[cpt_cands] + seg_costs
        F[tstar], tau = find_min(F_cost, pen)
        candidates[tstar] = cpt_cands[tau]

        ineq_prune = [val < F[tstar] for val in F_cost]
        R = [cpt_cands[j] for j, val in enumerate(ineq_prune) if val]
        R.append(tstar - 1)
        R = np.array(R, dtype=np.int)

    last = candidates[-1]
    changepoints = [last]
    while last > 0:
        last = candidates[last]
        changepoints.append(last)

    return sorted(changepoints)











size = 100

mean_a = 0.0
mean_b = 1.0
mean_c = 2.0
var = 0.1

data_a = np.random.normal(mean_a, var, size)
data_b = np.random.normal(mean_b, var, size)
data_c = np.random.normal(mean_c, var, size)
data = np.append(data_a, data_b)
data = np.append(data, data_c)

list_result = pelt(normal_mean(data, var), len(data))
[0, 100] # since data is random, sometimes it might be different, but most of the time there will be at most a couple more values around 100


plt.plot(data_a, label='a')
plt.plot(data_b, label='b')
plt.plot(data, label='data')
plt.legend()
plt.show()

print(list_result)


