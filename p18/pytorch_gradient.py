import torch
import numpy as np

########################################
# Proof of concept - "Working", somewhat

x1 = torch.tensor([1.,1.,1.], requires_grad=False).float()
x2 = torch.tensor([2.,2.,2.], requires_grad=False).float()
w = torch.tensor([0.,0.,0.], requires_grad=True).float()

f1 = torch.dot(x1, w) 
f2 = torch.dot(x2, w)

grads = torch.autograd.grad(f1, w)

####
# grads contains tensor([[ 1.,  1.,  1.], [ 2.,  2.,  2.]])
# which is the desired output, [∇w f1, ∇w f2]