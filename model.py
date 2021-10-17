import torch.nn as nn
import torch.nn.functional as F
import torch
    
from torchvision import models

class ImageClassificator(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.soft = nn.Softmax(dim = 1)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.soft(x)
    
    
def get_restnet152():
    
    model = models.resnet152(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 2),
                                    nn.LogSoftmax(dim=1))
    
    return model

def get_device():
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")