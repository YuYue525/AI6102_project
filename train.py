import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from glob import glob
from collections import defaultdict

import torchvision
from random import choice
from torch.utils.data import Dataset
from PIL import Image

class KinDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = list(person_to_images_map.keys())

    def __len__(self):
        return len(self.relations) * 2
               
    def __getitem__(self, idx):
        
        if (idx%2==0):
            p1, p2 = self.relations[idx//2]
            label = 1
        else:
            while True:
                p1 = choice(self.ppl)
                p2 = choice(self.ppl)
                if p1 != p2 and (p1, p2) not in self.relations and (p2, p1) not in self.relations:
                    break 
            label = 0
        
        path1, path2 = choice(self.person_to_images_map[p1]), choice(self.person_to_images_map[p2])
        img1, img2 = Image.open(path1), Image.open(path2)
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        
        return img1, img2, label

train_file_path = './recognizing-faces-in-the-wild/train_relationships.csv'
train_folders_path = './recognizing-faces-in-the-wild/train/'
test_folders_path = './recognizing-faces-in-the-wild/test/'
valid_folders_name = 'F08'

all_images = glob(train_folders_path + "*/*/*.jpg")

import os

if (os.path.exists("log.txt")):
    os.remove("log.txt")

train_valid_images = all_images
train_images = [x for x in all_images if valid_folders_name not in x]
valid_images = [x for x in all_images if valid_folders_name in x]

train_valid_images_map = defaultdict(list)
train_images_map = defaultdict(list)
valid_images_map = defaultdict(list)

for x in train_valid_images:
    train_valid_images_map[x.split('/')[-3] + "/" + x.split('/')[-2]].append(x)

for x in train_images:
    train_images_map[x.split('/')[-3] + "/" + x.split('/')[-2]].append(x)

for x in valid_images:
    valid_images_map[x.split('/')[-3] + "/" + x.split('/')[-2]].append(x)

relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))

family_member = [x.split('/')[-3] + '/' + x.split('/')[-2] for x in all_images]
relationships = [x for x in relationships if x[0] in family_member and x[1] in family_member]

train_valid = relationships
train = [x for x in relationships if valid_folders_name not in x[0]]
valid = [x for x in relationships if valid_folders_name in x[0]]

from torch.utils.data import DataLoader
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) 
])

valid_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) 
])

trainset = KinDataset(train, train_images_map, train_transform)
validset = KinDataset(valid, valid_images_map, valid_transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
validloader = DataLoader(validset, batch_size=128, shuffle=False)

from facenet_pytorch import InceptionResnetV1

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = InceptionResnetV1(pretrained='vggface2')
        
        emb_len = 512
        self.last = nn.Sequential(
            nn.Linear(4*emb_len, 200, bias=False),
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, input1, input2):
        
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)
        
        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
        x2 = torch.pow(emb1 - emb2, 2)
        x3 = emb1 * emb2
        x4 = emb1 + emb2
        
        x = torch.cat((x1,x2,x3,x4), dim=1)
        x = self.last(x)
        
        return x

def train():
    net.train()
    train_loss = 0.0
    running_loss = 0.0
    running_corrects = 0
    
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1,1).to(device)
        output = net(img1, img2)
        preds = output > 0.5
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == (label>0.5))
        
        step = 100
        if i % step == step-1:
            print(' [{} - {:.2f}%],\ttrain loss: {:.5}'.format(epoch+1, 100*(i+1)/len(trainloader), running_loss/step/200))
            running_loss = 0
        
    train_loss /= len(trainset)
    running_corrects = running_corrects.item()/len(trainset)
    f = open("log.txt", "a+")
    f.write('[{}], \ttrain loss: {:.5}\tacc: {:.5}\n'.format(epoch+1, train_loss, running_corrects))
    f.close()
    return train_loss, running_corrects

def validate():
    net.eval()
    val_loss = 0.0
    running_corrects = 0
    
    for batch in validloader:
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1,1).to(device)
        with torch.no_grad():
            output = net(img1, img2)
            preds = output>0.5
            loss = criterion(output, label)
            
        val_loss += loss.item()
        running_corrects += torch.sum(preds == (label>0.5))
    
    val_loss /= len(validset)
    running_corrects = running_corrects.item()/len(validset)
    f = open("log.txt", "a+")
    f.write('[{}], \tval loss: {:.5}\tacc: {:.5}\n'.format(epoch+1, val_loss, running_corrects))
    f.close()

    return val_loss, running_corrects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_model = 10
nets = []

for model_i in n_model:

    net = SiameseNet().to(device)
    
    nets.append(net)

    lr = 1e-3

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, patience=10)

    num_epoch = 200

    best_val_loss = 1000
    best_epoch = 0

    history = []
    accuracy = []

    for epoch in range(num_epoch):
        train_loss, train_acc = train()
        val_loss, val_acc = validate()
        history.append((train_loss, val_loss))
        accuracy.append((train_acc, val_acc))
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.torch.save(net.state_dict(), 'net_checkpoint_'+str(model_i)+'.pth')

    torch.save(net.state_dict(), 'net_full_training_'+str(model_i)+'.pth')
'''
import matplotlib.pyplot as plt

plt.figure()
plt.title("loss curves")
plt.plot(range(num_epoch), [x[0] for x in history], label="train loss")
plt.plot(range(num_epoch), [x[1] for x in history], label="valid loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.png")

plt.figure()
plt.title("accuracy curves")
plt.plot(range(num_epoch), [x[0] for x in accuracy], label="train acc")
plt.plot(range(num_epoch), [x[1] for x in accuracy], label="valid acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("acc.png")
'''

