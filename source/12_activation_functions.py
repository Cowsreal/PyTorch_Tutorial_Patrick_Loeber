#activation functions introduce non linearity inside our neural network
#Most popular activation functions
#1. Step Function
#       Not really used...
#2. Sigmoid
#       Outputs 0->1, usually used as the last layer in binary classification
#3. TanH
#       Outputs -1->1, used in hidden layers
#4. ReLU
#       If x<0, ReLU(x) = 0, otherwise, ReLU(x) = x, most popular choice of hidden layers
#5. Leaky ReLU
#       If x>=0, f(x) = x, otherwise, f(x) = ax for some a in R, improved version of ReLU, tries to solve vanishing gradient problem
#           If you notice that your weights aren't updating, try using leaky ReLU instead of ReLU
#6. Softmax
#       Last layer in multi class classification problems

import torch
import torch.nn as nn
import torch.nn.functional as F

#Example:

#option 1

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


#option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out