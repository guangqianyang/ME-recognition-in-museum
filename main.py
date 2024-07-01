import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision  
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data_loader import Imageloader
from network import ApexNet
import os
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

device = torch.device('cuda:0')

image_root = 'optical_data/'
batch_size = 64
num_workers = 4
imglist = os.listdir(image_root)
train_data = Imageloader(48, image_root, imglist)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


#weights = "weights/micro_best.pth"
net = ApexNet(4).to(device)

max_epoch = 80

optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0, last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss()

log_dir = './log_dir_128_31'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

for epoch in range(max_epoch):
    train_loss = 0

    labels = []
    y_scores = []
    net.train()
    predictions = []
    print('Training---------------->')

    for image1, image2, image3, label in train_loader:
        
        image1 = image1.to(device)
        image2 = image2.to(device)
        image3 = image3.to(device)

        label = label.to(device)

        pred = net(image1, image2, image3)

        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedular.step()
        train_loss += loss.item()
        predictions.extend(torch.argmax(pred, dim=1).cpu().numpy())
        labels.extend(label.cpu().numpy())
        softmax_outputs = nn.functional.softmax(pred, dim=1)
        y_scores.extend(softmax_outputs.cpu().detach().numpy())

    trainloss = train_loss/len(train_loader)
    train_acc = accuracy_score(labels, predictions)
    train_f1_score = f1_score(labels,predictions,average='macro')
    labels_new = np.zeros((len(labels), 4))
    for i, value in enumerate(labels):
        labels_new[i, value] = 1
    train_auc = roc_auc_score(labels_new, y_scores, multi_class='ovo')
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Acc/train', train_acc, epoch)
    writer.add_scalar('F1 socre/train', train_f1_score, epoch)
    writer.add_scalar('AUC/train', train_auc, epoch)
    writer.add_scalar('Learning rate/train', optimizer.param_groups[0]['lr'], epoch)
    print('Epoch: {} | train_loss: {:.6f} '.format(epoch, trainloss))
    if epoch%10==0:
        torch.save(net.state_dict(), 'weights/micro_{}.pth'.format(epoch))

torch.save(net.state_dict(), 'weights/micro_final.pth')
