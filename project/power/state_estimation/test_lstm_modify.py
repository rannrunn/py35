import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

epochs = 1000
output_size = 1
learning_rate = 0.01
beta1 = 0.5
beta2 = 0.999

input_size = 9
hidden_size = 10
batch_size = 20

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size=hidden_size
        self.lin = nn.Linear( input_size+hidden_size , 4*hidden_size )

    def forward(self, x, state0):
        h0,c0=state0
        x_and_h0 = torch.cat((x,h0), 1)
        u=self.lin(x_and_h0)
        i=F.sigmoid( u[ : , 0*self.hidden_size : 1*self.hidden_size ] )
        f=F.sigmoid( u[ : , 1*self.hidden_size : 2*self.hidden_size ] )
        g=F.tanh(    u[ : , 2*self.hidden_size : 3*self.hidden_size ] )
        o=F.sigmoid( u[ : , 3*self.hidden_size : 4*self.hidden_size ] )
        c= f*c0 + i*g
        h= o*F.tanh(c)
        return (h,c)

# construct LSTM Cell
rnn = LSTMCell(input_size, hidden_size)
rnn.cuda()

a = np.arange(0.01, 10.01, 0.01)
b = a.reshape(-1, 10)

# generate fake data
h0=torch.rand(batch_size,hidden_size).cuda()
c0=torch.rand(batch_size,hidden_size).cuda()
hh0=Variable(h0,requires_grad=True)
cc0=Variable(c0,requires_grad=True)

mceloss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, betas=(beta1, beta2))

# run the cell 1000 times forward and backward
t0=time.time()
for epoch in range(epochs):
    for batch in range(0, len(a), batch_size):
        data = torch.Tensor(b[batch:batch + 20,:]).cuda()
        xx=Variable(data,requires_grad=True)

        optimizer.zero_grad()
        for _ in range(input_size):
            print(xx.size())
            print(hh0.size())
            print(cc0.size())
            hh0,cc0=rnn(xx, (hh0,cc0)  )
        loss = mceloss(hh0, hh0)
        loss.backward()
        optimizer.step()

print('time in s : '+ str(time.time()-t0) )




