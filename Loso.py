import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision  
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data_loader import Imageloader
from network import generate_model
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, recall_score, precision_score
import argparse
import random
parser = argparse.ArgumentParser()

### Model S/L
parser.add_argument("--save_path", type=str, default="weights")
parser.add_argument("--model_type", type=str, default="vit_B", help="Which model to use (vit_L, vit_B, etc.)")
### Training
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--data_type", type=str, default='MME')
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--workers", type=int, default="18")
parser.add_argument("--data_parallel", action='store_true', default=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.data_type == 'MME':
    image_root = 'complete_data/'
elif args.data_type == 'realMME':
    image_root = 'realMME/'
elif args.data_type == 'LabMME':
    image_root = 'optical_data/'
else:
    exit(f'dataset {args.data_type} doesn\'t exist')
batch_size = 128
num_workers = 12

imglist = os.listdir(image_root)
labels = []
y_scores = []
predictions = []

print('{0} will be loaded with using dataset {1}'.format(args.model_type, args.data_type))


def lock_random_seed(seed) : 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

lock_random_seed(args.seed)

for i in tqdm(range(len(imglist))):
    test_list = [imglist[i]]
    # generate the train data list after remove the i-th element
    train_list = imglist[:i] + imglist[i+1:]
        
    train_data = Imageloader(48, image_root, train_list, args.data_type)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_data = Imageloader(48, image_root, test_list, args.data_type)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    #weights = "weights/micro_best.pth"
    net = generate_model(args.model_type)
    max_epoch = args.epochs

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(max_epoch):
        train_loss = 0

        for image1, image2, image3, label in train_loader:
            
            image1 = image1.cuda()
            image2 = image2.cuda()
            image3 = image3.cuda()

            label = label.cuda()

            pred = net(image1, image2, image3)

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedular.step()
            train_loss += loss.item()

    # test the sample
    with torch.no_grad():
        net.eval()

        for image1, image2, image3, label in test_loader:
            image1 = image1.cuda()
            image2 = image2.cuda()
            image3 = image3.cuda()

            label = label.cuda()

            pred = net(image1, image2, image3)
            predictions.extend(torch.argmax(pred, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())
            softmax_outputs = nn.functional.softmax(pred, dim=1)
            y_scores.extend(softmax_outputs.cpu().detach().numpy())



#print('predictions is: ', predictions)
#print('labels is: ', labels)

data = pd.DataFrame({'predictions':predictions, 'labels':labels})
data.to_csv('results/predictions_{0}_{1}.csv'.format(args.model_type, args.data_type), index=False)
val_f1_score = f1_score(labels,predictions,average='macro')
labels_new = np.zeros((len(labels), 4))
for i, value in enumerate(labels):
    labels_new[i, value] = 1
auc = roc_auc_score(labels_new, y_scores, multi_class='ovo')
val_recall = recall_score(labels, predictions,zero_division = 0,average='macro')
val_precision = precision_score(labels,predictions, zero_division = 0,average='macro')
val_acc = accuracy_score(labels, predictions)

print('precision is: ', val_precision)
print('recall is: ', val_recall)
print('f1 score is: ', val_f1_score)
print('acc is: ', val_acc)
print('auc is: ', auc)


