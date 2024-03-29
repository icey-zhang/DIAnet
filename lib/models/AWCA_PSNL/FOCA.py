import torch
from torch import nn

class FOCA(nn.Module):
    def __init__(self,inchannel):
        super(FOCA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1))

        self.fc = nn.Sequential(
      
            nn.Linear(512,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,512),
            nn.Sigmoid(),
         
         )
       
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.gap(x).view(batch_size, 512, -1)
        x = x.permute(0, 2, 1) #将tensor的维度换位,b*(hw)*c
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        x = x.permute(0, 2, 1).contiguous()
        #print(x.shape)
        x = x.view(batch_size, 512, 1,1)
      
       
        #x = torch.reshape(x, (N, 492 * 492)
        return x
