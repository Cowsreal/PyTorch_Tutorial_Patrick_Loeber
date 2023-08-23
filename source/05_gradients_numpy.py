#We'll first implement everything manually, and slowly but surely, use Torch's automation features to help us
#STEP 1:
#Prediction: Manually
#Gradients Computation: Manually
#Loss Computation: Manually
#Parameter Updates: Manually

import numpy as np

#f = w * x, no bias
#Let's let w = 2

X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)

w = 0.0

#Model Prediction
def forward(x):
    return w * x

#Loss
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#Gradient
#Mean Squared Error = 1/N * (w * x - y)**2
#dJ/dw = 1/N * 2 * (w * x - y) * x
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean() #.mean() since we divide the result by N

print(f"Prediction before training: f(5) = {forward(5):.3f}")

#Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)
    
    #loss
    l = loss(Y, y_pred)
    
    #gradients
    dw = gradient(X, Y, y_pred)
    
    #update weights
    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")