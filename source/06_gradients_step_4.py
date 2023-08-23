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

X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #Define layers
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

#Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(X)
    
    #loss
    l = loss(Y, y_pred)

    #Compute gradients
    l.backward()
    
    #update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")