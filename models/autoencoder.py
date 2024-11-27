
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        in_dim = 784
        self.fc1 = nn.Linear(in_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 30)
        self.fc5 = nn.Linear(30, 250)
        self.fc6 = nn.Linear(250, 500)
        self.fc7 = nn.Linear(500, 1000)
        self.fc8 = nn.Linear(1000, in_dim)
        self.r = F.relu
   
    def forward(self, inputs):
        # encoder
        x = inputs.view(inputs.size(0), -1)
        x = self.r(self.fc1(x))
        x = self.r(self.fc2(x))
        x = self.r(self.fc3(x))
        x = self.fc4(x)
        # decoder
        x = self.r(self.fc5(x))
        x = self.r(self.fc6(x))
        x = self.r(self.fc7(x))
        x = self.fc8(x)
        return x
        #return x.view(-1, 1, 28, 28)