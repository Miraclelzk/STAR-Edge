import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class DescClassifier(nn.Module):
    def __init__(self):
        super(DescClassifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(10, 32),  
            nn.ReLU(),
            nn.Linear(32, 64),  
            nn.ReLU(),
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.Linear(32, 1),    
            nn.Sigmoid()         
        )

    def forward(self, x):
        # print('x', x.shape)
        out = self.mlp(x).view(-1)
        # print('out', out.shape)
        return out
    
    def load(self, model_file):
        checkpoint = torch.load(model_file)
        self.load_state_dict(checkpoint['state_dict'])

if __name__ == '__main__':
    met = torch.rand(32, 10)
    b, c= met.shape
    net = DescClassifier()
    output = net(met)
    print(output.shape)
