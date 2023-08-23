#We'll first implement everything manually, and slowly but surely, use Torch's automation features to help us
#STEP 1:
#Prediction: Manually
#Gradients Computation: Autograd
#Loss Computation: PyTorch loss
#Parameter Updates: PyTorch optimizer

#1) Design model(input, output size, forward pass)
#2) Construct loss and optimizer
#3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn

#f = w * x, no bias
#Let's let w = 2

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, requires_grad = True)

#Model Prediction
def forward(x):
    return w * x


print(f"Prediction before training: f(5) = {forward(5):.3f}")

#Training
learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)
    
    #loss
    l = loss(Y, y_pred)

    #Compute gradients
    l.backward()
    
    #update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch % 1 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")