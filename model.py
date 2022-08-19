import torch
import torch.nn as nn 
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #Output size formula: [(side + 2 x padding - kernel size)/stride] + 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=0) #32-3/1 +1 = 30 x 30 x 32
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding=0) #30-3/1 +1 = 28 x 28 x 32
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2) #Output = 28-2/2+1 = 14 x 14 x 32  #27 x 27 x 32
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=0) # 14-3/1+1 = 12 x 12 x 64        
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=0) #12-3/1+1 = 10 x 10 x 64
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) #Output: 10-2/2+1 = 5 x 5 x 64

        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.final = nn.Linear(512,10)

    def forward(self, x):
        
        #First convolutional layer
        out = self.conv1(x)
        out = F.relu(out)
        
        #Second convolutional layer
        out = self.conv2(out)
        out = F.relu(out)
        
        #First pooling layer
        out = self.pool1(out)
        
        #Third convolutional layer
        out = self.conv3(out)
        out = F.relu(out)
        
        #Last convolutional layer
        out = self.conv4(out)
        out = F.relu(out)
        
        #Second pooling layer
        out = self.pool2(out)
        
        #Fully-connected layers 
        out = torch.flatten(out,1)
        out = self.fc1(out)
        out = F.relu(out)

        out = self.final(out)
        return out
