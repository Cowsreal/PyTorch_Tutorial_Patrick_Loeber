#Autograd tutorial https://www.youtube.com/watch?v=c36lUUr864M

import torch

#Tensor with 3 random values
x = torch.randn(3, requires_grad = True)
print(x)

y = x + 2
#Since requires_grad = True, y has attribute grad_fn
print(y)
#grad_fn = <AddBackward0>

z = y * y * 2
print(z)
#grad_fn = <MulBackward0>

z = z.mean()
print(z)
#grad_fn = <MeanBackward0>

z.backward() #Calculates dz/dx
print(x.grad)

#If the output is nonscalar... then

z = y * y * 2

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)

#Suppose we want to tell PyTorch to stop tracking the backwards gradient on x...

#x.requires_grad_(False)
#y = x.detach()     Outputs a new reassignable variable with require_grad = False
#with torch.no_grad():
#   ...

#IMPORTANT: WHEN RUNNING FOR LOOPS, IF .backward() IS CALLED, GRADIENTS ARE ACCUMULATED VIA SUMMATION

weights = torch.ones(4, requires_grad = True)

for epochs in range(1):
    output = (weights*3).sum()
    output.backward()
    print(weights.grad)
    
for epochs in range(1):
    output = (weights*3).sum()
    output.backward()
    print(weights.grad)

for epochs in range(1):
    output = (weights*3).sum()
    output.backward()
    print(weights.grad)
    
#Since we kept doing the same operation, the gradients are still stored in memory and each iteration simply sums

#Correct way:

weights = torch.ones(4, requires_grad = True)

for epochs in range(3):
    output = (weights*3).sum()
    output.backward()
    print(weights.grad)
    weights.grad.zero_()