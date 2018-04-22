# http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
# https://github.com/jcjohnson/pytorch-examples#pytorch-control-flow--weight-sharing

# In this implementation we implement our own custom autograd function to perform the ReLU function.

import torch
from torch.autograd import Variable


"""
In PyTorch 
we can easily define our own autograd operator 
by defining a subclass of torch.autograd.Function

and

implementing the forward and backward functions. 

We can then use our new autograd operator by constructing an instance 
and calling it like a function, passing Variables containing input data.

# https://github.com/jcjohnson/pytorch-examples#pytorch-control-flow--weight-sharing
"""


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # todo: stach...? cache...?
        ctx.save_for_backward(input)        # todo: ctx 가 뭐지...?
        # a context object that can be used
        # to stash information for backward computation. You can cache arbitrary
        # objects for use in the backward pass using the ctx.save_for_backward method.
        # todo: save_for_backward( ) 가 뭐지...?
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors      # todo: ctx.saved_tensors 가 뭐지...?
        grad_input = grad_output.clone()    # todo: clone 이 뭐지...?
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
epochs = 500
for epoch in range(epochs):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations on Variables; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(epoch, loss.data[0])

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()