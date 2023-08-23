import torch

#Linear regression

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

#Forward pass and compute loss

y_hat = x * w

loss = (y_hat - y)**2

print(loss)

#Backward pass, the local gradients and backward pass chain rule is computed automatically for us, no work in our end
loss.backward()
print(w.grad)

#Update weights,
#Repeat...

