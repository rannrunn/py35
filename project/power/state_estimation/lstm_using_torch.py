import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

epochs = 2000
learning_rate = 0.001
hidden = 1
input = 1
output = 1


Wf = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Wi = Variable(torch.Tensor([0]), requires_grad=True).cuda()
WC = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Wo = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Wy = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Bf = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Bi = Variable(torch.Tensor([0]), requires_grad=True).cuda()
BC = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Bo = Variable(torch.Tensor([0]), requires_grad=True).cuda()
By = Variable(torch.Tensor([0]), requires_grad=True).cuda()
Wf.retain_grad()
Wi.retain_grad()
WC.retain_grad()
Wo.retain_grad()
Wy.retain_grad()
Bf.retain_grad()
Bi.retain_grad()
BC.retain_grad()
Bo.retain_grad()
By.retain_grad()


list_num = np.arange(1, 101, 1)

beta1 = 0.5
beta2 = 0.999
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))

for epoch in range(epochs):

    Xt = 0

    HtB = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    Ht = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    HtA = Variable(torch.Tensor([0]), requires_grad=True).cuda()

    CtB = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    Ct = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    CtA = Variable(torch.Tensor([0]), requires_grad=True).cuda()

    forget_gate = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    input_gate = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    Cp = Variable(torch.Tensor([0]), requires_grad=True).cuda()
    output_gate = Variable(torch.Tensor([0]), requires_grad=True).cuda()

    HtB.retain_grad()
    Ht.retain_grad()
    HtA.retain_grad()

    CtB.retain_grad()
    Ct.retain_grad()
    CtA.retain_grad()

    forget_gate.retain_grad()
    input_gate.retain_grad()
    Cp.retain_grad()
    output_gate.retain_grad()

    for idx, item in enumerate(list_num[:-1]):

        Xt = Variable(torch.Tensor([list_num[idx]]), requires_grad=True).cuda()
        XtA = Variable(torch.Tensor([list_num[idx + 1]]), requires_grad=True).cuda()

        CtB = CtA
        HtB = HtA

        # forget gate
        forget_gate = F.sigmoid(HtB * Wf + Xt * Wf + Bf)
        # input_gate
        input_gate = F.sigmoid(HtB * Wi + Xt * Wi + Bi)
        # premature C
        Cp = F.tanh(HtB * WC + Xt * WC + BC)
        # output_gate
        output_gate = F.sigmoid(HtB * Wo + Xt * Wi + Bo)

        CtA = forget_gate * CtB + input_gate * Cp
        HtA = F.tanh(CtA) * output_gate

        loss = (XtA - HtA).pow(2)
        loss.backward()



    print('Epoch:', epoch, 'HtA:', HtA)














