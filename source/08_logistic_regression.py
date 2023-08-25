import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0) prepare the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape[0], X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#1) model
# f = wx + b, sigmoid activation function
class logisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(logisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = logisticRegression(n_features)

#2) loss and optimizer
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#3) training loop
num_epochs = 2000

for epoch in range(num_epochs):
    #forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    #backward pass
    loss.backward()
    
    #updates
    optimizer.step()
    
    #zero the gradients
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")

#validation step using test data
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()
    acc = y_pred_class.eq(y_test).sum() / y_test.shape[0]
    print(f"Accuracy = {acc:.4f}")