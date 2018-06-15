import numpy as np
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
plt.style.use('seaborn-white')
import time

# Read and process data

data = open('./input.txt', 'r').read()



chars = list(set(data))
data_size, X_size = len(data), len(chars)
print("data has %d characters, %d unique" % (data_size, X_size))
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

print(char_to_idx)
print(type(idx_to_char))
print(data)
print(chars)
# Parameters

H_size = 100 # Size of the hidden layer
T_steps = 25 # Number of time steps (length of the sequence) used for training
learning_rate = 1e-1 # Learning rate
weight_sd = 0.1 # Standard deviation of weights for initialization
z_size = H_size + X_size # Size of concatenate(H, X) vector


# Activations and derivs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y


# Init weights

W_f = np.random.randn(H_size, z_size) * weight_sd + 0.5
b_f = np.zeros((H_size, 1))

W_i = np.random.randn(H_size, z_size) * weight_sd + 0.5
b_i = np.zeros((H_size, 1))

W_C = np.random.randn(H_size, z_size) * weight_sd
b_C = np.zeros((H_size, 1))

W_o = np.random.randn(H_size, z_size) * weight_sd + 0.5
b_o = np.zeros((H_size, 1))

#For final layer to predict the next character
W_y = np.random.randn(X_size, H_size) * weight_sd
b_y = np.zeros((X_size, 1))


# Gradients

dW_f = np.zeros_like(W_f)
dW_i = np.zeros_like(W_i)
dW_C = np.zeros_like(W_C)

dW_o = np.zeros_like(W_o)
dW_y = np.zeros_like(W_y)

db_f = np.zeros_like(b_f)
db_i = np.zeros_like(b_i)
db_C = np.zeros_like(b_C)

db_o = np.zeros_like(b_o)
db_y = np.zeros_like(b_y)


def forward(x, h_prev, C_prev):
    assert x.shape == (X_size, 1)
    assert h_prev.shape == (H_size, 1)
    assert C_prev.shape == (H_size, 1)

    z = np.row_stack((h_prev, x))
    f = sigmoid(np.dot(W_f, z) + b_f)
    i = sigmoid(np.dot(W_i, z) + b_i)
    C_bar = tanh(np.dot(W_C, z) + b_C)

    C = f * C_prev + i * C_bar
    o = sigmoid(np.dot(W_o, z) + b_o)
    h = o * tanh(C)

    y = np.dot(W_y, h) + b_y
    p = np.exp(y) / np.sum(np.exp(y))

    return z, f, i, C_bar, C, o, h, y, p

def backward(target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, y, p):

    global dW_f, dW_i, dW_C, dW_o, dW_y
    global db_f, db_i, db_C, db_o, db_y

    assert z.shape == (X_size + H_size, 1)
    assert y.shape == (X_size, 1)
    assert p.shape == (X_size, 1)

    for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
        assert param.shape == (H_size, 1)

    dy = np.copy(p)
    print('dy_shape', dy.shape)
    # print('dy', dy)
    dy[target] -= 1
    print('dy2_shape', dy.shape)
    # print('dy2', dy)

    dW_y += np.dot(dy, h.T)
    db_y += dy

    print('dW_y_shape:', dW_y.shape)
    # print('dW_y:', dW_y)
    print('h.T_shape', h.T.shape)
    # print('h.T', h.T)
    print('db_y_shape:', db_y.shape)
    # print('db_y:', db_y)

    dh = np.dot(W_y.T, dy)
    dh += dh_next
    do = dh * tanh(C)
    do = dsigmoid(o) * do
    dW_o += np.dot(do, z.T)
    db_o += do

    print('dh_shape:', dh.shape)
    # print('dh:', dh)
    print('do_shape:', do.shape)
    # print('do:', do)
    print('dW_o_shape:', dW_o.shape)
    # print('dW_o:', dW_o)
    print('db_o_shape:', db_o.shape)
    # print('db_o:', db_o)

    dC = np.copy(dC_next)
    dC += dh * o * dtanh(tanh(C))
    dC_bar = dC * i
    dC_bar = dC_bar * dtanh(C_bar)
    dW_C += np.dot(dC_bar, z.T)
    db_C += dC_bar

    print('dC_shape:', dC.shape)
    # print('dC:', dC)
    print('dC_bar_shape:', dC_bar.shape)
    # print('dC_bar:', dC_bar)
    print('dW_C_shape:', dW_C.shape)
    # print('dW_C:', dW_C)
    print('db_C_shape:', db_C.shape)
    # print('db_C:', db_C)

    di = dC * C_bar
    di = dsigmoid(i) * di
    dW_i += np.dot(di, z.T)
    db_i += di

    print('di_shape:', di.shape)
    # print('di:', di)
    print('dW_i_shape:', dW_i.shape)
    # print('dW_i:', dW_i)
    print('db_i_shape:', db_i.shape)
    # print('db_i:', db_i)

    df = dC * C_prev
    df = dsigmoid(f) * df
    dW_f += np.dot(df, z.T)
    db_f += df

    print('df_shape:', df.shape)
    # print('df:', df)
    print('dW_f_shape:', dW_f.shape)
    # print('dW_f:', dW_f)
    print('db_f_shape:', db_f.shape)
    # print('db_f:', db_f)

    dz = np.dot(W_f.T, df) \
         + np.dot(W_i.T, di) \
         + np.dot(W_C.T, dC_bar) \
         + np.dot(W_o.T, do)
    dh_prev = dz[:H_size, :]
    dC_prev = f * dC

    print('dh_prev_shape:', dh_prev.shape)
    # print('dh_prev:', dh_prev)
    print('dC_prev_shape:', dC_prev.shape)
    # print('dC_prev:', dC_prev)
    print('dz_shape:', dz.shape)
    # print('dz:', dz)

    print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')

    return dh_prev, dC_prev

def forward_backward(inputs, targets, h_prev, C_prev):
    # To store the values for each time step
    x_s, z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, y_s, p_s = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    # Values at t - 1
    h_s[-1] = np.copy(h_prev)
    C_s[-1] = np.copy(C_prev)

    loss = 0
    # Loop through time steps
    assert len(inputs) == T_steps
    for t in range(len(inputs)):
        x_s[t] = np.zeros((X_size, 1))

        x_s[t][inputs[t]] = 1 # Input character
        z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], y_s[t], p_s[t] \
            = forward(x_s[t], h_s[t - 1], C_s[t - 1]) # Forward pass
        loss += -np.log(p_s[t][targets[t], 0]) # Loss for at t


    for dparam in [dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y]:
        dparam.fill(0)

    dh_next = np.zeros_like(h_s[0]) #dh from the next character
    dC_next = np.zeros_like(C_s[0]) #dh from the next character

    for t in reversed(range(len(inputs))):
        # Backward pass
        dh_next, dC_next = backward(target = targets[t], dh_next = dh_next, dC_next = dC_next, C_prev = C_s[t-1],
                                    z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t], C = C_s[t], o = o_s[t],
                                    h = h_s[t], y = y_s[t], p = p_s[t])

    # Clip gradients to mitigate exploding gradients
    for dparam in [dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y]:
        np.clip(dparam, -1, 1, out=dparam)

    return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]

def sample(h_prev, C_prev, first_char_idx, sentence_length):
    x = np.zeros((X_size, 1))
    x[first_char_idx] = 1

    h = h_prev
    C = C_prev

    indexes = []

    for t in range(sentence_length):
        _, _, _, _, C, _, h, _, p = forward(x, h, C)
        idx = np.random.choice(range(X_size), p=p.ravel())
        x = np.zeros((X_size, 1))
        x[idx] = 1
        indexes.append(idx)

    return indexes

def update_status(inputs, h_prev, C_prev):
    #initialized later
    global plot_iter, plot_loss
    global smooth_loss

    # Get predictions for 200 letters with current model

    sample_idx = sample(h_prev, C_prev, inputs[0], 2000)
    txt = ''.join(idx_to_char[idx] for idx in sample_idx)

    #Print prediction and loss
    print("----\n %s \n----" % (txt, ))
    print("iter %d, loss %f" % (iteration, smooth_loss))


# Memory variables for Adagrad

mW_f = np.zeros_like(W_f)
mW_i = np.zeros_like(W_i)
mW_C = np.zeros_like(W_C)
mW_o = np.zeros_like(W_o)
mW_y = np.zeros_like(W_y)

mb_f = np.zeros_like(b_f)
mb_i = np.zeros_like(b_i)
mb_C = np.zeros_like(b_C)
mb_o = np.zeros_like(b_o)
mb_y = np.zeros_like(b_y)


# Exponential average of loss
# Initialize to a error of a random model
smooth_loss = -np.log(1.0 / X_size) * T_steps

iteration, p = 0, 0

# For the graph
plot_iter = np.zeros((0))
plot_loss = np.zeros((0))


while True:
    # Try catch for interruption
    try:
        # Reset
        if p + T_steps >= len(data) or iteration == 0:
            g_h_prev = np.zeros((H_size, 1))
            g_C_prev = np.zeros((H_size, 1))
            p = 0


        inputs = [char_to_idx[ch] for ch in data[p: p + T_steps]]
        targets = [char_to_idx[ch] for ch in data[p + 1: p + T_steps + 1]]
        #        print("INPUTS typ({}) len({}) ins({})".format(type(inputs), len(inputs), inputs))

        loss, g_h_prev, g_C_prev =  forward_backward(inputs, targets, g_h_prev, g_C_prev)
        time.sleep(100000)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Print every hundred steps
        if iteration % 100 == 0:
              (inputs, g_h_prev, g_C_prev)


        # Update weights
        for param, dparam, mem in zip([W_f, W_i, W_C, W_o, W_y, b_f, b_i, b_C, b_o, b_y],
                                      [dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y],
                                      [mW_f, mW_i, mW_C, mW_o, mW_y, mb_f, mb_i, mb_C, mb_o, mb_y]):
            mem += dparam * dparam # Calculate sum of gradients
            #print(learning_rate * dparam)
            param += -(learning_rate * dparam / np.sqrt(mem + 1e-8))

        plot_iter = np.append(plot_iter, [iteration])
        plot_loss = np.append(plot_loss, [loss])

        p += T_steps
        iteration += 1
    except KeyboardInterrupt:
        update_status(inputs, g_h_prev, g_C_prev)
        plt.show()
        break
