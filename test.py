import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from glob import glob
from collections import defaultdict

import torchvision
from random import choice
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

batch_size = 1

submission_path = "./recognizing-faces-in-the-wild/sample_submission.csv"
img_root_dir = "./recognizing-faces-in-the-wild/test/"
sample_submission = pd.read_csv(submission_path)

new = sample_submission["img_pair"].str.split("-", n = 1, expand = True)

# making separate first name column from new data frame
sample_submission["Person1"]= new[0]
# making separate last name column from new data frame
sample_submission["Person2"]= new[1]

class FamilyTestDataset(Dataset):
    """Family Dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.relations)
    
    def __getpair__(self,idx):
        pair = os.path.join(self.root_dir, self.relations.iloc[idx,2]), os.path.join(self.root_dir, self.relations.iloc[idx,3])
        return pair
    
    def __getlabel__(self,idx):
        return self.relations.iloc[idx,4]
    
    def __getitem__(self, idx):
        try:
            pair =  self.__getpair__(idx)

            img0 = Image.open(pair[0])
            img1 = Image.open(pair[1])

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
            
            return idx, img0, img1
            
        except Exception as e:
            print(e)
            
test_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

test_dataset = FamilyTestDataset(df = sample_submission, root_dir = img_root_dir, transform = test_transform)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size)

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
        
n_model = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_i in range(n_model):

    net = SiameseNet().to(device)
    net.load_state_dict(torch.load('net_checkpoint_'+str(model_i)+'.pth'))
    
    net.eval()
    for i, data in enumerate(test_loader, 0):
        row, img0, img1 = data
        row, img0, img1 = row.cuda(), img0.cuda(), img1.cuda()
       
        output = torch.sigmoid(net(img0, img1))

        sample_submission.loc[row.item(),'is_related'] = output[0].item()
        
    sample_submission.drop(columns =["Person1","Person2"], inplace = True)
    
    sample_submission.to_csv('output.csv',index=False)
        
