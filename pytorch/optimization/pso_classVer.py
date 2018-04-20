# https://gist.github.com/btbytes/79877
# https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
# https://www.researchgate.net/post/What_are_the_best_PSO_parameter_values

"""
PSO
"""

import random
import numpy as np


# todo: NN structure 최대한 자유롭게 PSO 가 설정하도록 바꿈
# todo: act 나 linear 순서도 자유롭게... 자유롭게 할 수 있는 부분은 최대한 자유롭게 설정하게함


class PsoHyperParam:
    def __init__(self, inertia_, c1_, c2_, population_size_):
        self.inertia = inertia_
        self.c1 = c1_
        self.c2 = c2_
        self.population_size = population_size_


class Particle:
    def __init__(self, dim_, loss_fn_, get_init_param_fn):
        self.dim = dim_
        self.param = get_init_param_fn()
        self.p_loss = loss_fn_(self.param)  # p: particle
        self.l_param = self.param  # l: local best (individual best)
        self.l_loss = loss_fn_(self.l_param)
        self.direction = np.array([0 for _ in range(self.dim)])

    def update_param(self, hp, g_param):
        # g_param: parameter of global best particle
        # hp: PsoHyperParam class
        r1 = random.random()
        r2 = random.random()
        social_term = hp.c2 * r2 * (g_param - self.param)
        cognitive_term = hp.c1 * r1 * (self.l_param - self.param)

        self.direction = hp.inertia * self.direction + social_term + cognitive_term
        self.param = self.param + self.direction
        return self.param


def get_opt_param(hp, loss_fn, get_init_param_fn, is_terminated_fn, verbose=False):
    # make particles
    particles = [None for _ in range(hp.population_size)]

    for idx in range(hp.population_size):
        particles[idx] = Particle(dim_=col_num, loss_fn_=loss_fn, get_init_param_fn=get_init_param_fn)

    # initialize global particle
    g = Particle(dim_=col_num, loss_fn_=loss_fn, get_init_param_fn=get_init_param_fn)
    g.l_loss = loss_fn(g.param)

    count = 0
    while True:
        # optimize particle
        for p in particles:
            # evaluate loss of particle
            p.param = np.array(list(map(int, p.param)))  # to int
            loss = loss_fn(p.param)

            # update 'local' loss & param
            if loss < p.l_loss:
                p.l_loss = loss
                p.l_param = p.param

            # update 'global' loss & param
            if loss < g.l_loss:
                g.l_loss = loss
                g.param = p.param

            if verbose is True:
                print("\ncount:   ", count)
                print("g loss:  ", g.l_loss)
                print("p loss:  ", loss)
                print("p param: ", p.param, "\n")

            count += 1

            # update particle
            p.update_param(hp=hp, g_param=g.param)

            # stop condition
            if is_terminated_fn(count):
                break

        # stop condition
        if is_terminated_fn(count):
            break

    return g


if __name__ == "__main__":
    def loss_fn(args):
        _sum = 0
        for val in args:
            _sum += abs(val)
        return _sum


    def get_init_param(min_=0, max_=100, dim=2):
        _params = [None for _ in range(dim)]
        for _idx in range(dim):
            _params[_idx] = random.randint(min_, max_)
        return np.array(_params)


    def is_terminated(count):
        if count > 300:
            return True
        else:
            return False

    # init
    col_num = 2

    # set hyper parameter
    hp = PsoHyperParam(inertia_=0.4, c1_=2, c2_=2, population_size_=10)
    global_param = get_opt_param(hp=hp, loss_fn=loss_fn, get_init_param_fn=get_init_param,
                                 is_terminated_fn=is_terminated, verbose=True)

    print(global_param.param)
    print(global_param.l_loss)
