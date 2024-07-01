import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial

# convoluation block consist 2 3*3 convolution layer with relu activation function and batch normalization, the first convoluaiton layer has a stide of 1, the 
# second one has a stride of 2




class ApexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_class = num_classes
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Block1 = nn.Sequential(nn.Linear(2304+2304, 1024), nn.ReLU(inplace=True))
        self.Block2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.Block3 = nn.Sequential(nn.Linear(1024, num_classes))


    def forward(self, x1, x2, x3):
        x1 = self.conv1_1(x1)   # x1: [batch_size, 1, 48, 48]  
        x1 = self.maxpool(x1)   # x1: [batch_size, 6, 24, 24]  
        x1 = self.conv1_2(x1)   # x1: [batch_size, 16, 24, 24]
        x1 = self.maxpool(x1)   # x1: [batch_size, 16, 12, 12]

        x2 = self.conv2_1(x2)   # X2: [batch_size, 1, 48, 48]
        x2 = self.maxpool(x2)   # X2: [batch_size, 6, 24, 24]
        x2 = self.conv2_2(x2)   # X2: [batch_size, 16, 24, 24]
        x2 = self.maxpool(x2)   # x2: [batch_size, 16, 12, 12]

        x1 = x1.view(-1, 2304)
        x2 = x2.view(-1, 2304)

        out = torch.cat([x1,x2], dim=1)
        out = self.Block1(out)
        out = self.Block2(out)
        out = self.Block3(out)

        return out



class block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class proposed_v1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_class = num_classes
        self.Block1_1 = block(1, 8)
        self.Block1_2 = block(8,16)
        self.Block2_1 = block(1, 8)
        self.Block2_2 = block(8, 16)
        self.Block3_1 = block(1, 8)
        self.Block3_2 = block(8, 16)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True))

        self.Block1 = nn.Sequential(nn.Linear(128+128+128, 128), nn.ReLU(inplace=True))
        self.Block2 = nn.Sequential(nn.Linear(128, num_classes))


    def forward(self, x1, x2, x3):
        x1 = self.Block1_1(x1)
        x1 = self.Block1_2(x1)
        x2 = self.Block2_1(x2)
        x2 = self.Block2_2(x2)
        x3 = self.Block3_1(x3)
        x3 = self.Block3_2(x3)


        x1 = self.maxpool1(x1)
        x2 = self.maxpool2(x2)
        x3 = self.maxpool3(x3)

        x1 = x1.view(-1, 576)
        x2 = x2.view(-1, 576)
        x3 = x3.view(-1, 576)
        
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)


        out = torch.cat([x1, x2, x3], dim=1)

        out = self.Block1(out)
        out = self.Block2(out)

        return out



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class proposed_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_class = num_classes
        self.Block1_1 = block(1, 8)
        self.Block1_2 = block(8,16)
        self.Block2_1 = block(1, 8)
        self.Block2_2 = block(8, 16)
        self.Block3_1 = block(1, 8)
        self.Block3_2 = block(8, 16)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True))

        self.Block = nn.Sequential(nn.Linear(576, num_classes))
        self.cross_attention1 = CrossAttention(576)
        self.cross_attention2 = CrossAttention(576)
        self.cross_attention3 = CrossAttention(576)


    def forward(self, x1, x2, x3):
        x1 = self.Block1_1(x1)
        x1 = self.Block1_2(x1)
        x2 = self.Block2_1(x2)
        x2 = self.Block2_2(x2)
        x3 = self.Block3_1(x3)
        x3 = self.Block3_2(x3)


        x1 = self.maxpool1(x1)
        x2 = self.maxpool2(x2)
        x3 = self.maxpool3(x3)

        x1 = x1.view(-1, 576) # shape: [batch, 576]
        x2 = x2.view(-1, 576) # shape: [batch, 576]
        x3 = x3.view(-1, 576) # shape: [batch, 576]

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)

        context = torch.cat([x1, x2, x3], dim=1)
        x1 = self.cross_attention1(x1, context)
        x2 = self.cross_attention2(x2, context)
        x3 = self.cross_attention3(x3, context)

        #x1 = self.fc1(x1)
        #x2 = self.fc2(x2)
        #x3 = self.fc3(x3)
        
        out = torch.cat([x1, x2, x3], dim=1)
        out = torch.mean(out, dim=1)

        out = self.Block(out)



        return out


def generate_model(model_type):
    if model_type =='ApexNet':
        return ApexNet(num_classes=4).cuda()
    elif model_type == 'proposed_v1':
        return proposed_v1(num_classes=4).cuda()
    elif model_type=='proposed_v2':
        return proposed_v2(num_classes=4).cuda()
    else:
        exit('The model {model_type} is not existed')


