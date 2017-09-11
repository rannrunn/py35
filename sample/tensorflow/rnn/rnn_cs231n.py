# https://minjejeon.github.io/learningstock/2016/08/28/min-char-rnn-%ED%95%9C%EA%B8%80-%EC%A3%BC%ED%95%B4-3.html

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

import numpy as np
np.set_printoptions(threshold=np.nan)

# data I/O
data = open('input.txt', 'r').read()  # should be simple plain text file
print(data)
# set은 집합을 만드는 함수
chars = list(set(data))
print(chars)
data_size, vocab_size = len(data), len(chars)
print(data_size)
print(vocab_size)
#print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1
print(learning_rate)
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
print('Wxh')
print(Wxh)
print(Wxh.shape)
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
print('Whh')
# print(Whh)
print(Whh.shape)
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
print('Why')
# print(Why)
print(Why.shape)
bh = np.zeros((hidden_size, 1))  # hidden bias
print('bh')
# print(bh)
print(bh.shape)
by = np.zeros((vocab_size, 1))  # output bias
print('by')
# print(by)
print(by.shape)

def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    #print(hs)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        # softmax로 각 글자의 등장 가능성을 확률로 표시
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        # cross-entropy를 이용하여 정답과 비교하여 손실값 판정
        #print(ps[t][targets[t], 0])
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))): # forward pass의 과정을 반대로 진행(t=24부터 시작)
        dy = np.copy(ps[t])
        # y의 그래디언트 계산, softmax 함수의 그래디언트 계산
        dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        #print('xxxxxxx')
        #print(hs[t].shape)
        #print('yyyyyyy')
        #print(hs[t].T.shape)
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        # 그래디언트 발산 방지
        # 그래디언트의 각 원소에서 절대값이 5가 넘어가는 값은 그래디언트의 발산 방지를 위해 np.clip 함수를 이용하여 최대 절대값을 5로 만들어 주고 결과를 반환
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    print('sample')
    for t in range(n):
        # print(Wxh)
        # print(x)
        # print(np.dot(Wxh, x))
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        # print(y)
        # np.exp : 밑(base)이 자연상수 e 인 지수함수로 변환해준다.
        # print(np.sum(np.exp(y))) => 64.XXXXXXXXXX ~ 65.XXXXXXXXXX
        p = np.exp(y) / np.sum(np.exp(y))
        # print(p)
        # print(range(vocab_size))
        # print(p.ravel())
        # 샘플링. 임의성을 부여하기 위해 argmax대신 array p에서 주어진 확률에 의해 하나의 문자를 선택
        # ravel : 행렬의 1행부터 순차적으로 원소 값을 불러와서 1차원 array를 만드는 함수
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        # ix = np.random.choice(65, 1)
        # print(ix)
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
print(-np.log(1.0 / vocab_size))
print(smooth_loss)
num = 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        # print('----\n %s \n----' % (txt,))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        # print('iter %d, loss: %f' % (n, smooth_loss))  # print progress
        pass

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        #print(param.shape)
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    print('param')
    print(param.shape)
    print('dparam')
    print(dparam.shape)
    print('mem')
    print(mem.shape)

    p += seq_length  # move data pointer
    n += 1  # iteration counter

    xxxx = np.random.choice(5, 10, p=[0.0, 0, 0.1, 0.9, 0])
    print(xxxx)

    num += 1

    if num == 2:
        break
